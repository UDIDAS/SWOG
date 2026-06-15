// fastloops: graph-minor pooling on 2D images or 3D volumes.
// Contraction uses Kruskal-like canonical-anchored merging (SEMIR paper).
use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

const VISITED:                u8 = 0b0000_0001;
const NODE_MERGED:            u8 = 0b0000_0010;
const NODE_DELETED:           u8 = 0b0000_0100;
const NODE_ADJACENT_BOUNDARY: u8 = 0b0000_1000;
const EDGE_MERGED:            u8 = 0b0001_0000;
const EDGE_DELETED:           u8 = 0b0010_0000;
const IN_CURRENT_REGION:      u8 = 0b0100_0000;
const N_EDGE_FEATURES:     usize = 4;

const WR: u32 = 2126; const WG: u32 = 7152; const WB: u32 = 722;
#[inline(always)] fn luma_u8(rgb: [u8; 3]) -> u8 { ((rgb[0] as u32 * WR + rgb[1] as u32 * WG + rgb[2] as u32 * WB) / 10000) as u8 }

#[inline(always)]
fn channel_distance(a: &[u8], b: &[u8], method: u8, weighted_luma: bool) -> u32 {
    if weighted_luma {
        let la = luma_u8([a[0], a[1], a[2]]) as i32;
        let lb = luma_u8([b[0], b[1], b[2]]) as i32;
        return (la - lb).unsigned_abs();
    }
    match method {
        0 => a.iter().zip(b).map(|(x, y)| (*x as i16 - *y as i16).unsigned_abs() as u32).sum(),
        1 => {
            let s: u32 = a.iter().zip(b).map(|(x, y)| { let d = (*x as i16 - *y as i16).unsigned_abs() as u32; d * d }).sum();
            (s as f64).sqrt().round() as u32
        }
        2 => a.iter().zip(b).map(|(x, y)| (*x as i16 - *y as i16).unsigned_abs() as u32).max().unwrap_or(0),
        _ => unreachable!(),
    }
}

fn euclidean_gcd(mut a: usize, mut b: usize) -> usize { while b != 0 { let t = b; b = a % b; a = t; } a }
fn coprime_step(val: usize, divisor: f64) -> usize {
    if val <= 1 { return 1; }
    let split = (((val as f64) / divisor.max(1.0)).floor() as usize).clamp(1, val.saturating_sub(1));
    for i in split..val { if euclidean_gcd(val, i) == 1 { return i; } }
    for i in (1..split).rev() { if euclidean_gcd(val, i) == 1 { return i; } }
    1
}

struct Layout { nf: usize, area: usize, s: [usize; 3], cov: [usize; 6], chan0: usize, min_max: [usize; 6], boundary: usize, canon0: usize }

fn build_layout(d: usize, c: usize) -> Layout {
    if d == 2 {
        Layout { nf: 13 + c, area: 0, s: [1, 2, 0], cov: [3, 4, 5, 0, 0, 0], chan0: 6, min_max: [6+c, 6+c+1, 6+c+2, 6+c+3, 0, 0], boundary: 6+c+4, canon0: 6+c+5 }
    } else {
        Layout { nf: 20 + c, area: 0, s: [1, 2, 3], cov: [4, 5, 6, 7, 8, 9], chan0: 10, min_max: [10+c, 10+c+1, 10+c+2, 10+c+3, 10+c+4, 10+c+5], boundary: 10+c+6, canon0: 10+c+7 }
    }
}

#[inline(always)] fn flat<const D: usize>(coord: &[usize; D], stride: &[usize; D]) -> usize { (0..D).map(|k| coord[k] * stride[k]).sum() }

struct Out<const D: usize> { adj: Vec<u8>, ddims: [usize; D], node_feats: Vec<u64>, n_reg: usize, nf: usize, labels: Vec<i64>, edge_index: Vec<i64>, edge_feats: Vec<u64>, n_edges: usize }

fn process<const D: usize>(
    data: &[u8], dims: [usize; D], c: usize, merge_distance: u32, cut_distance: u32, delete_small: usize, delete_large: usize,
    value_min: u32, value_max: u32, _divisor: f64, method: u8, weighted_luma: bool, max_nonzero: usize
) -> Out<D> {
    let lay = build_layout(D, c); let nf = lay.nf;
    let n_voxels = dims.iter().product::<usize>();
    let mut ddims = [0usize; D]; for k in 0..D { ddims[k] = dims[k] * 2 - 1; }
    let mut dstride = [0usize; D]; let mut vstride = [0usize; D]; dstride[D-1] = 1; vstride[D-1] = 1;
    for k in (0..D-1).rev() { dstride[k] = dstride[k+1] * ddims[k+1]; vstride[k] = vstride[k+1] * dims[k+1]; }
    let n_adj = ddims.iter().product::<usize>(); let mut adj = vec![0u8; n_adj];

    // =========================================================================
    // Phase 1: Kruskal-like canonical-anchored edge contraction
    //
    // Paper: "Edge contraction merges neighboring supernodes v_i, v_k if
    // ||I_{v_i} - I_{v_k}|| <= ψ where I_v is the intensity of the canonical
    // voxel. Contraction recurses until no further valid merges exist."
    //
    // Implementation: bucket-sort all adjacent voxel pairs by distance,
    // process from smallest to largest. Union-Find tracks supernodes.
    // Before each merge, check canonical intensities (not edge distance).
    // The canonical anchors prevent transitive drift beyond ±ψ.
    // =========================================================================

    // Union-Find with path compression + union by size
    let mut uf_parent: Vec<usize> = (0..n_voxels).collect();
    let mut uf_size: Vec<u64> = vec![1; n_voxels];
    // Per-supernode canonical: intensity of the first-merged (most central) voxel
    // Initialize to each voxel's own intensity
    let mut canon_int: Vec<u32> = (0..n_voxels).map(|v| {
        (0..c).map(|ch| data[v * c + ch] as u32).sum::<u32>() / c as u32
    }).collect();

    #[inline]
    fn uf_find(x: usize, parent: &mut [usize]) -> usize {
        let mut r = x;
        while parent[r] != r { r = parent[r]; }
        let mut c = x;
        while parent[c] != r { let n = parent[c]; parent[c] = r; c = n; }
        r
    }

    // Bucket sort: collect all face-adjacent pairs by distance (0..=255)
    let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 256];
    for v in 0..n_voxels {
        let mut rem = v; let mut vc = [0usize; D];
        for k in (0..D).rev() { vc[k] = rem % dims[k]; rem /= dims[k]; }
        // Only check "forward" neighbors to avoid duplicates
        for axis in 0..D {
            if vc[axis] + 1 < dims[axis] {
                let mut nb = vc; nb[axis] += 1;
                let nbf = flat::<D>(&nb, &vstride);
                let dist = channel_distance(&data[v*c..v*c+c], &data[nbf*c..nbf*c+c], method, weighted_luma);
                if dist <= 255 {
                    buckets[dist as usize].push((v, nbf));
                }
            }
        }
    }

    // Process buckets 0..=merge_distance: merge pairs if canonicals are within ψ
    for dist in 0..=merge_distance.min(255) {
        for &(a, b) in &buckets[dist as usize] {
            let ra = uf_find(a, &mut uf_parent);
            let rb = uf_find(b, &mut uf_parent);
            if ra == rb { continue; }

            // SEMIR check: compare canonical intensities, not edge distance
            let canon_diff = (canon_int[ra] as i32 - canon_int[rb] as i32).unsigned_abs();
            if canon_diff > merge_distance { continue; }

            // Merge smaller into larger; canonical = larger's canonical
            let (big, small) = if uf_size[ra] >= uf_size[rb] { (ra, rb) } else { (rb, ra) };
            uf_parent[small] = big;
            uf_size[big] += uf_size[small];
            // Canonical stays as the larger supernode's (already set)

            // Mark edge as merged in adj tensor
            let mut va = [0usize; D]; let mut vb = [0usize; D];
            let mut rem = a; for k in (0..D).rev() { va[k] = rem % dims[k]; rem /= dims[k]; }
            rem = b; for k in (0..D).rev() { vb[k] = rem % dims[k]; rem /= dims[k]; }
            let mut ec = [0usize; D]; for k in 0..D { ec[k] = va[k] + vb[k]; } // doubled coords: edge midpoint
            let ef = flat::<D>(&ec, &dstride);
            adj[ef] |= EDGE_MERGED;
        }
    }

    // Build contiguous labels from Union-Find
    let mut root_to_id: HashMap<usize, i64> = HashMap::new();
    let mut labels = vec![-1i64; n_voxels];
    let mut cur: usize = 0;
    for v in 0..n_voxels {
        let r = uf_find(v, &mut uf_parent);
        if let Some(&id) = root_to_id.get(&r) {
            labels[v] = id;
        } else {
            root_to_id.insert(r, cur as i64);
            labels[v] = cur as i64;
            cur += 1;
        }
    }

    // Mark node/edge states in adj tensor for boundary detection
    for v in 0..n_voxels {
        let mut rem = v; let mut vc = [0usize; D];
        for k in (0..D).rev() { vc[k] = rem % dims[k]; rem /= dims[k]; }
        let dc = { let mut c2 = [0usize; D]; for k in 0..D { c2[k] = vc[k] * 2; } c2 };
        let cf = flat::<D>(&dc, &dstride);
        adj[cf] |= VISITED | NODE_MERGED;
    }

    // =========================================================================
    // Phase 2: Node deletion (same as Luke's original)
    // =========================================================================
    // Compute per-supernode stats for deletion decisions
    let mut sn_area = vec![0u64; cur];
    let mut sn_chan_sum = vec![0u64; cur];
    for v in 0..n_voxels {
        let r = labels[v] as usize;
        sn_area[r] += 1;
        let val: u64 = (0..c).map(|ch| data[v * c + ch] as u64).sum();
        sn_chan_sum[r] += val;
    }

    let mut keep = vec![true; cur];
    for r in 0..cur {
        let area = sn_area[r] as usize;
        let mean = (sn_chan_sum[r] / (sn_area[r].max(1) * c as u64)) as u32;
        let size_ok = (area > delete_small && area < delete_large) || (delete_small >= delete_large);
        let value_ok = mean >= value_min && mean <= value_max;
        if !size_ok || !value_ok { keep[r] = false; }
    }

    // Remap: deleted supernodes get label -1, survivors get contiguous IDs
    let mut remap = vec![-1i64; cur];
    let mut new_cur: usize = 0;
    for r in 0..cur {
        if keep[r] { remap[r] = new_cur as i64; new_cur += 1; }
    }
    for v in 0..n_voxels {
        let old = labels[v];
        if old >= 0 {
            let new_label = remap[old as usize];
            labels[v] = new_label;
            if new_label < 0 {
                // Mark deleted in adj
                let mut rem = v; let mut vc = [0usize; D];
                for k in (0..D).rev() { vc[k] = rem % dims[k]; rem /= dims[k]; }
                let dc = { let mut c2 = [0usize; D]; for k in 0..D { c2[k] = vc[k] * 2; } c2 };
                adj[flat::<D>(&dc, &dstride)] |= NODE_DELETED;
            }
        }
    }
    cur = new_cur;

    // =========================================================================
    // Phase 3: Compute node features for surviving supernodes
    // =========================================================================
    let mut node_feats = vec![0u64; cur * nf];
    for r in 0..cur {
        let base = r * nf;
        node_feats[base + lay.min_max[0]] = u64::MAX;
        node_feats[base + lay.min_max[2]] = u64::MAX;
        if D == 3 { node_feats[base + lay.min_max[4]] = u64::MAX; }
        for k in 0..D { node_feats[base + lay.canon0 + k] = u64::MAX; }
    }
    for v in 0..n_voxels {
        let r = labels[v]; if r < 0 { continue; }
        let r = r as usize; let base = r * nf;
        let mut rem = v; let mut vc = [0usize; D];
        for k in (0..D).rev() { vc[k] = rem % dims[k]; rem /= dims[k]; }
        let x = vc[D-1] as u64; let y = vc[D-2] as u64;
        node_feats[base + lay.area] += 1;
        node_feats[base + lay.s[0]] += x; node_feats[base + lay.s[1]] += y;
        node_feats[base + lay.cov[0]] += x*x; node_feats[base + lay.cov[1]] += y*y; node_feats[base + lay.cov[2]] += x*y;
        if x < node_feats[base + lay.min_max[0]] { node_feats[base + lay.min_max[0]] = x; }
        if x > node_feats[base + lay.min_max[1]] { node_feats[base + lay.min_max[1]] = x; }
        if y < node_feats[base + lay.min_max[2]] { node_feats[base + lay.min_max[2]] = y; }
        if y > node_feats[base + lay.min_max[3]] { node_feats[base + lay.min_max[3]] = y; }
        if D == 3 {
            let z = vc[0] as u64; node_feats[base + lay.s[2]] += z;
            node_feats[base + lay.cov[3]] += z*z; node_feats[base + lay.cov[4]] += x*z; node_feats[base + lay.cov[5]] += y*z;
            if z < node_feats[base + lay.min_max[4]] { node_feats[base + lay.min_max[4]] = z; }
            if z > node_feats[base + lay.min_max[5]] { node_feats[base + lay.min_max[5]] = z; }
        }
        for ch in 0..c { node_feats[base + lay.chan0 + ch] += data[v * c + ch] as u64; }
        { let mut less = false; for k in 0..D { let cv = node_feats[base + lay.canon0 + k]; let vk = vc[k] as u64;
            if vk < cv { less = true; break; } if vk > cv { break; } }
            if less { for k in 0..D { node_feats[base + lay.canon0 + k] = vc[k] as u64; } } }
        // Boundary detection
        let mut on_boundary = false;
        for axis in 0..D {
            for dir in [-1isize, 1] {
                let nk = vc[axis] as isize + dir;
                if nk < 0 || nk >= dims[axis] as isize { on_boundary = true; continue; }
                let mut nb = vc; nb[axis] = nk as usize;
                let nbf = flat::<D>(&nb, &vstride);
                if labels[nbf] < 0 || labels[nbf] != r as i64 { on_boundary = true; }
            }
        }
        if on_boundary { node_feats[base + lay.boundary] += 1; }
    }
    let n_reg = cur;

    // =========================================================================
    // Phase 4: Edge building (Luke's original — cut edges kept with cut_frac)
    // =========================================================================
    let total = 3usize.pow(D as u32); let mut voff = Vec::new();
    for code in 0..total {
        let mut t = code; let mut d = [0isize; D]; let mut nz = 0;
        for k in 0..D { let trit = t % 3; t /= 3; d[k] = match trit { 0 => 0, 1 => 1, _ => -1 }; if d[k] != 0 { nz += 1; } }
        if nz == 0 || nz > max_nonzero { continue; }
        let mut canon = true; for k in 0..D { if d[k] != 0 { canon = d[k] > 0; break; } } if canon { voff.push(d); }
    }
    let mut emap = HashMap::new();
    for p in 0..n_voxels {
        let la = labels[p]; if la < 0 { continue; }
        let mut rem = p; let mut vc = [0usize; D]; for k in (0..D).rev() { vc[k] = rem % dims[k]; rem /= dims[k]; }
        for off in &voff {
            let mut nb = [0usize; D]; let mut ok = true;
            for k in 0..D { let q = vc[k] as isize + off[k]; if q < 0 || q >= dims[k] as isize { ok = false; break; } nb[k] = q as usize; }
            if !ok { continue; }
            let nbf = flat::<D>(&nb, &vstride); let lb = labels[nbf]; if lb < 0 || lb == la { continue; }
            let key = if la < lb { (la, lb) } else { (lb, la) };
            let dist = channel_distance(&data[p*c..p*c+c], &data[nbf*c..nbf*c+c], method, weighted_luma);
            let mut es = [0usize; D]; for k in 0..D { es[k] = (2 * vc[k] as isize + off[k]) as usize; }
            let ef = flat::<D>(&es, &dstride);
            let is_cut = dist >= cut_distance; if is_cut { adj[ef] |= EDGE_DELETED; }
            let e = emap.entry(key).or_insert([0u64; N_EDGE_FEATURES]);
            e[0] += 1; e[1] += dist as u64; if dist as u64 > e[2] { e[2] = dist as u64; } e[3] += is_cut as u64;
        }
    }
    let mut keys: Vec<(i64, i64)> = emap.keys().copied().collect(); keys.sort_unstable();
    let n_edges = keys.len();
    let mut edge_index = vec![0i64; 2 * n_edges]; let mut edge_feats = vec![0u64; N_EDGE_FEATURES * n_edges];
    for (i, k) in keys.iter().enumerate() {
        edge_index[i] = k.0; edge_index[n_edges + i] = k.1;
        let e = emap[k]; for j in 0..N_EDGE_FEATURES { edge_feats[i * N_EDGE_FEATURES + j] = e[j]; }
    }
    Out { adj, ddims, node_feats, n_reg, nf, labels, edge_index, edge_feats, n_edges }
}

#[pyfunction]
#[pyo3(signature = (image, merge_distance=None, cut_distance=None, delete_small_node_max_size=None, delete_large_node_min_size=None, delete_value_min=None, delete_value_max=None, iterator_divisor=None, distance_method=None, use_weighted_luminance=None, connectivity=None))]
fn merge_and_cut<'py>(
    py: Python<'py>, image: PyReadonlyArrayDyn<'_, u8>, merge_distance: Option<u32>, cut_distance: Option<u32>, delete_small_node_max_size: Option<usize>, delete_large_node_min_size: Option<usize>,
    delete_value_min: Option<u32>, delete_value_max: Option<u32>, iterator_divisor: Option<f64>, distance_method: Option<&str>, use_weighted_luminance: Option<bool>, connectivity: Option<&str>
) -> PyResult<(Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<u64>>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let view = image.as_array(); let ndim = view.ndim(); if ndim != 3 && ndim != 4 { return Err(PyValueError::new_err("image must be (H,W,C) or (D,H,W,C) channel-last u8")); }
    let shape = view.shape().to_vec(); let c = shape[ndim - 1]; if c < 1 { return Err(PyValueError::new_err("channel axis must be >= 1")); }
    let method = match distance_method.unwrap_or("chebyshev") { "manhattan" => 0, "euclidean" => 1, "chebyshev" => 2, _ => return Err(PyValueError::new_err("invalid distance_method")) };
    let weighted_luma = use_weighted_luminance.unwrap_or(false); if weighted_luma && c != 3 { return Err(PyValueError::new_err("weighted luma requires RGB")); }
    let max_nonzero = match connectivity.unwrap_or("faces") { "faces" => 1, "faces_edges" => 2, "faces_corners" => 3, _ => return Err(PyValueError::new_err("invalid connectivity")) };
    let divisor = iterator_divisor.unwrap_or(4.12).max(1.0); let merge_distance = merge_distance.unwrap_or(10); let cut_distance = cut_distance.unwrap_or(100);
    let value_min = delete_value_min.unwrap_or(0); let value_max = delete_value_max.unwrap_or(u32::MAX);
    let data = view.as_slice().ok_or_else(|| PyValueError::new_err("image must be C-contiguous"))?;
    if ndim == 3 {
        let dims = [shape[0], shape[1]]; let n = dims[0] * dims[1]; let delete_small = delete_small_node_max_size.unwrap_or((n as f64).log2() as usize); let delete_large = delete_large_node_min_size.unwrap_or((n as f64).powf(0.8) as usize);
        let out = py.allow_threads(|| process::<2>(data, dims, c, merge_distance, cut_distance, delete_small, delete_large, value_min, value_max, divisor, method, weighted_luma, max_nonzero));
        let adj_arr = Array2::from_shape_vec((out.ddims[0], out.ddims[1]), out.adj).unwrap(); let lab_arr = Array2::from_shape_vec((dims[0], dims[1]), out.labels).unwrap();
        let feat_arr = Array2::from_shape_vec((out.n_reg, out.nf), out.node_feats).unwrap(); let ei_arr = Array2::from_shape_vec((2, out.n_edges), out.edge_index).unwrap(); let ef_arr = Array2::from_shape_vec((out.n_edges, N_EDGE_FEATURES), out.edge_feats).unwrap();
        Ok((PyArray2::from_owned_array_bound(py, feat_arr), PyArray2::from_owned_array_bound(py, ei_arr), PyArray2::from_owned_array_bound(py, ef_arr), PyArray2::from_owned_array_bound(py, lab_arr).into_any(), PyArray2::from_owned_array_bound(py, adj_arr).into_any()))
    } else {
        let dims = [shape[0], shape[1], shape[2]]; let n = dims[0] * dims[1] * dims[2]; let delete_small = delete_small_node_max_size.unwrap_or((n as f64).log2() as usize); let delete_large = delete_large_node_min_size.unwrap_or((n as f64).powf(0.8) as usize);
        let out = py.allow_threads(|| process::<3>(data, dims, c, merge_distance, cut_distance, delete_small, delete_large, value_min, value_max, divisor, method, weighted_luma, max_nonzero));
        let adj_arr = Array3::from_shape_vec((out.ddims[0], out.ddims[1], out.ddims[2]), out.adj).unwrap(); let lab_arr = Array3::from_shape_vec((dims[0], dims[1], dims[2]), out.labels).unwrap();
        let feat_arr = Array2::from_shape_vec((out.n_reg, out.nf), out.node_feats).unwrap(); let ei_arr = Array2::from_shape_vec((2, out.n_edges), out.edge_index).unwrap(); let ef_arr = Array2::from_shape_vec((out.n_edges, N_EDGE_FEATURES), out.edge_feats).unwrap();
        Ok((PyArray2::from_owned_array_bound(py, feat_arr), PyArray2::from_owned_array_bound(py, ei_arr), PyArray2::from_owned_array_bound(py, ef_arr), PyArray3::from_owned_array_bound(py, lab_arr).into_any(), PyArray3::from_owned_array_bound(py, adj_arr).into_any()))
    }
}

#[pymodule] fn fastloops(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_function(wrap_pyfunction!(merge_and_cut, m)?)?; Ok(()) }
