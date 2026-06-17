// fastloops_band: full graph build by LEARNED INTENSITY BANDS (Stage-1 substrate).
//
// Floods a 3D grayscale volume by a learned band map (band_of: intensity -> band id),
// emitting one moment-carrying supernode per band-connected region plus the region
// adjacency graph. This is the substrate the D2 contraction learner trains on.
//
// v0.2.0: applies Stage-1 INTENSITY DELETION. A `deleted` mask (256,) flags intensities
// whose voxels are dropped from the representation entirely -- the sweep certified them as
// carrying negligible foreground (band_fg / GT <= deletion budget). A deleted voxel is
// invisible: it never seeds a region, is never absorbed into one, contributes no boundary
// length, gets label = -1, and produces no edge. The emitted tensor therefore matches the
// foreground prevalence the sweep certified. (With no deletions passed, behavior is
// identical to v0.1.0 and labels are all >= 0.)
//
// Differences from merge_and_cut (v0.3.0), by design:
//   * MERGE PREDICATE is band membership, not channel distance:
//       neighbor joins seed's region iff band_of[seed_I] == band_of[neighbor_I]
//       AND neither seed nor neighbor intensity is deleted.
//   * NO learned node deletion here beyond the certified-junk intensity delete above.
//   * NO cut_distance / cut_count. Edge features are [boundary_len, sum_dist, max_dist].
//   * MASK REQUIRED. Each node carries n_fg / n_bg voxel counts.
//
// CONTRACT (channel-last, C-contiguous, u8). 3D only; predicate keys on channel 0.
//   data    : (D, H, W, C) u8   -- volume
//   mask    : (D, H, W)   u8     -- foreground {0,1}
//   band_of : (256,)      u8     -- intensity -> band id
//   deleted : (256,)      u8     -- nonzero => intensity is deleted (dropped from graph)
//
// RETURNS (5-tuple):
//   node_features (N, 23)  u64  -- per surviving supernode
//   edge_index    (2, E)   i64  -- undirected, canonical (row0<row1), columns ref node rows
//   edge_features (E, 3)   u64  -- [boundary_len, sum_dist, max_dist]
//   labels        (D,H,W)  i64  -- voxel -> node id; -1 for deleted (and any non-node) voxels
//   adj           doubled  u8   -- cell complex (reprojection substrate)
//
// NODE FEATURE LAYOUT (3D, C=1):
//   0 area; 1,2,3 sx,sy,sz; 4..9 sxx,syy,szz,sxy,sxz,syz; 10 chan0 sum;
//   11..16 minx,maxx,miny,maxy,minz,maxz; 17 boundary; 18,19,20 canon[z,y,x];
//   21,22 n_fg,n_bg  -> nf = 23
//
// adj bitflags: VISITED 1, NODE_MERGED 2, NODE_ADJACENT_BOUNDARY 8,
//   EDGE_MERGED 16 (intra-region), EDGE_BOUNDARY 32 (inter-region), IN_CURRENT_REGION 64

use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArrayDyn, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

const VISITED:                u8 = 0b0000_0001;
const NODE_MERGED:            u8 = 0b0000_0010;
const NODE_ADJACENT_BOUNDARY: u8 = 0b0000_1000;
const EDGE_MERGED:            u8 = 0b0001_0000;
const EDGE_BOUNDARY:          u8 = 0b0010_0000;
const IN_CURRENT_REGION:      u8 = 0b0100_0000;

const N_EDGE_FEATURES: usize = 3;  // boundary_len, sum_dist, max_dist

fn euclidean_gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn coprime_step(val: usize, divisor: f64) -> usize {
    if val <= 1 { return 1; }
    let split = (((val as f64) / divisor.max(1.0)).floor() as usize)
        .clamp(1, val.saturating_sub(1));
    for i in split..val { if euclidean_gcd(val, i) == 1 { return i; } }
    for i in (1..split).rev() { if euclidean_gcd(val, i) == 1 { return i; } }
    1
}

struct Layout {
    nf: usize,
    area: usize,
    sx: usize, sy: usize, sz: usize,
    sxx: usize, syy: usize, szz: usize, sxy: usize, sxz: usize, syz: usize,
    chan0: usize,
    minx: usize, maxx: usize, miny: usize, maxy: usize, minz: usize, maxz: usize,
    boundary: usize,
    canon0: usize,
    n_fg: usize, n_bg: usize,
}

fn build_layout(c: usize) -> Layout {
    let chan0 = 10;
    let minx = chan0 + c;
    let boundary = minx + 6;
    let canon0 = boundary + 1;
    let n_fg = canon0 + 3;
    Layout {
        area: 0, sx: 1, sy: 2, sz: 3,
        sxx: 4, syy: 5, szz: 6, sxy: 7, sxz: 8, syz: 9,
        chan0,
        minx, maxx: minx + 1, miny: minx + 2, maxy: minx + 3, minz: minx + 4, maxz: minx + 5,
        boundary,
        canon0,
        n_fg, n_bg: n_fg + 1,
        nf: n_fg + 2,
    }
}

#[inline(always)]
fn flat3(coord: &[usize; 3], stride: &[usize; 3]) -> usize {
    coord[0] * stride[0] + coord[1] * stride[1] + coord[2] * stride[2]
}

struct Out {
    adj: Vec<u8>,
    ddims: [usize; 3],
    node_feats: Vec<u64>,
    n_reg: usize,
    nf: usize,
    labels: Vec<i64>,
    edge_index: Vec<i64>,
    edge_feats: Vec<u64>,
    n_edges: usize,
}

fn process(
    data: &[u8],
    mask: &[u8],
    band_of: &[u8],
    deleted: &[u8],
    dims: [usize; 3],
    c: usize,
    divisor: f64,
) -> Out {
    let lay = build_layout(c);
    let nf = lay.nf;

    let n_voxels = dims[0] * dims[1] * dims[2];
    let mut ddims = [0usize; 3];
    for k in 0..3 { ddims[k] = dims[k] * 2 - 1; }

    let mut dstride = [0usize; 3];
    let mut vstride = [0usize; 3];
    dstride[2] = 1; vstride[2] = 1;
    dstride[1] = ddims[2]; vstride[1] = dims[2];
    dstride[0] = ddims[1] * ddims[2]; vstride[0] = dims[1] * dims[2];

    let n_adj: usize = ddims[0] * ddims[1] * ddims[2];
    let mut adj = vec![0u8; n_adj];
    let mut labels = vec![-1i64; n_voxels];

    let mut step = [0usize; 3];
    let mut start = [0usize; 3];
    for k in 0..3 { step[k] = coprime_step(dims[k], divisor); start[k] = step[k] % dims[k]; }

    let offsets: [[isize; 3]; 6] = [
        [2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2],
    ];

    #[inline(always)]
    fn is_deleted(deleted: &[u8], data: &[u8], vf: usize, c: usize) -> bool {
        deleted[data[vf * c] as usize] != 0
    }

    let mut node_feats: Vec<u64> = Vec::new();
    let mut cur: usize = 0;
    let mut stack: Vec<[usize; 3]> = Vec::new();
    let mut region_voxels: Vec<usize> = Vec::new();

    // ---- PASS 1: band flood -> regions, node moments, fg/bg, intra-region edges ----
    for idx in 0..n_voxels {
        let mut rem = idx;
        let mut coord = [0usize; 3];
        let mut vseed = [0usize; 3];
        for k in (0..3).rev() {
            let i_k = rem % dims[k]; rem /= dims[k];
            let ck = (start[k] + i_k * step[k]) % dims[k];
            vseed[k] = ck; coord[k] = ck * 2;
        }
        let seed_flat = flat3(&coord, &dstride);
        if adj[seed_flat] & VISITED != 0 { continue; }

        let seed_voxel_flat = flat3(&vseed, &vstride);

        // DELETED seed: mark visited so we never reconsider it, emit no node, label stays -1.
        if is_deleted(deleted, data, seed_voxel_flat, c) {
            adj[seed_flat] |= VISITED;
            continue;
        }

        let base = cur * nf;
        if node_feats.len() < base + nf { node_feats.resize(base + nf, 0); }
        else { for t in base..base + nf { node_feats[t] = 0; } }
        node_feats[base + lay.minx] = u64::MAX;
        node_feats[base + lay.miny] = u64::MAX;
        node_feats[base + lay.minz] = u64::MAX;
        for k in 0..3 { node_feats[base + lay.canon0 + k] = u64::MAX; }

        let seed_band = band_of[data[seed_voxel_flat * c] as usize];

        adj[seed_flat] |= NODE_MERGED | IN_CURRENT_REGION;
        stack.clear();
        region_voxels.clear();
        stack.push(coord);

        while let Some(dc) = stack.pop() {
            let cf = flat3(&dc, &dstride);
            if adj[cf] & VISITED != 0 { continue; }
            adj[cf] |= VISITED;

            let mut vc = [0usize; 3];
            for k in 0..3 { vc[k] = dc[k] / 2; }
            let vf = flat3(&vc, &vstride);
            region_voxels.push(vf);
            let here = &data[vf * c..vf * c + c];

            let x = vc[2] as u64; let y = vc[1] as u64; let z = vc[0] as u64;
            node_feats[base + lay.area] += 1;
            node_feats[base + lay.sx] += x; node_feats[base + lay.sy] += y; node_feats[base + lay.sz] += z;
            node_feats[base + lay.sxx] += x*x; node_feats[base + lay.syy] += y*y; node_feats[base + lay.szz] += z*z;
            node_feats[base + lay.sxy] += x*y; node_feats[base + lay.sxz] += x*z; node_feats[base + lay.syz] += y*z;
            if x < node_feats[base+lay.minx] { node_feats[base+lay.minx]=x; }
            if x > node_feats[base+lay.maxx] { node_feats[base+lay.maxx]=x; }
            if y < node_feats[base+lay.miny] { node_feats[base+lay.miny]=y; }
            if y > node_feats[base+lay.maxy] { node_feats[base+lay.maxy]=y; }
            if z < node_feats[base+lay.minz] { node_feats[base+lay.minz]=z; }
            if z > node_feats[base+lay.maxz] { node_feats[base+lay.maxz]=z; }
            for ch in 0..c { node_feats[base + lay.chan0 + ch] += here[ch] as u64; }

            if mask[vf] != 0 { node_feats[base + lay.n_fg] += 1; }
            else { node_feats[base + lay.n_bg] += 1; }

            {
                let mut less = false;
                for k in 0..3 {
                    let cv = node_feats[base + lay.canon0 + k]; let v = vc[k] as u64;
                    if v < cv { less = true; break; } if v > cv { break; }
                }
                if less { for k in 0..3 { node_feats[base + lay.canon0 + k] = vc[k] as u64; } }
            }

            let mut on_boundary = false;
            for off in &offsets {
                let mut ncoord = [0usize; 3];
                let mut ecoord = [0usize; 3];
                let mut ok = true;
                for k in 0..3 {
                    let nv = dc[k] as isize + off[k];
                    if nv < 0 || nv >= ddims[k] as isize { ok = false; break; }
                    ncoord[k] = nv as usize;
                    ecoord[k] = (dc[k] as isize + off[k] / 2) as usize;
                }
                if !ok { continue; }
                let nf_flat = flat3(&ncoord, &dstride);
                let ef_flat = flat3(&ecoord, &dstride);

                let mut nvc = [0usize; 3];
                for k in 0..3 { nvc[k] = ncoord[k] / 2; }
                let nvf = flat3(&nvc, &vstride);

                // DELETED neighbor: invisible. Checked FIRST -- before any adj-flag branch --
                // so a deleted voxel that was itself seed-marked VISITED still cannot induce
                // boundary or absorption on a live region. Skip entirely.
                if is_deleted(deleted, data, nvf, c) { continue; }

                let n_flags = adj[nf_flat];
                if n_flags & IN_CURRENT_REGION != 0 { adj[ef_flat] |= EDGE_MERGED; continue; }
                if n_flags & (VISITED | NODE_MERGED) != 0 { on_boundary = true; continue; }

                let neigh_band = band_of[data[nvf * c] as usize];
                if neigh_band == seed_band {
                    adj[nf_flat] |= NODE_MERGED | IN_CURRENT_REGION;
                    adj[ef_flat] |= EDGE_MERGED;
                    stack.push(ncoord);
                } else {
                    on_boundary = true;
                }
            }
            if on_boundary {
                adj[cf] |= NODE_ADJACENT_BOUNDARY;
                node_feats[base + lay.boundary] += 1;
            }
        }

        for &df in region_voxels.iter() {
            let mut vc = [0usize; 3];
            let mut r = df;
            vc[0] = r / vstride[0]; r %= vstride[0];
            vc[1] = r / vstride[1]; vc[2] = r % vstride[1];
            let dcoord = [vc[0]*2, vc[1]*2, vc[2]*2];
            let dfl = flat3(&dcoord, &dstride);
            adj[dfl] &= !IN_CURRENT_REGION;
            labels[df] = cur as i64;
        }
        cur += 1;
    }

    let n_reg = cur;
    node_feats.truncate(n_reg * nf);

    // ---- PASS 2: region adjacency + edge features ----
    // Deleted voxels have labels = -1, so they are skipped as edge sources (la < 0) and as
    // edge targets (lb < 0). No deleted-intensity adjacency enters the edge set.
    let voff: [[isize; 3]; 3] = [[1,0,0],[0,1,0],[0,0,1]];
    let mut emap: HashMap<(i64, i64), [u64; N_EDGE_FEATURES]> = HashMap::new();
    for p in 0..n_voxels {
        let la = labels[p];
        if la < 0 { continue; }
        let mut rem = p; let mut vc = [0usize; 3];
        vc[0] = rem / vstride[0]; rem %= vstride[0];
        vc[1] = rem / vstride[1]; vc[2] = rem % vstride[1];
        for off in &voff {
            let mut nb = [0usize; 3]; let mut ok = true;
            for k in 0..3 {
                let q = vc[k] as isize + off[k];
                if q < 0 || q >= dims[k] as isize { ok = false; break; }
                nb[k] = q as usize;
            }
            if !ok { continue; }
            let nbf = flat3(&nb, &vstride);
            let lb = labels[nbf];
            if lb < 0 || lb == la { continue; }
            let key = if la < lb { (la, lb) } else { (lb, la) };
            let mut dist: u64 = 0;
            for ch in 0..c {
                let a = data[p*c+ch] as i32; let b = data[nbf*c+ch] as i32;
                dist += (a - b).unsigned_abs() as u64;
            }
            let mut es = [0usize; 3];
            for k in 0..3 { es[k] = (2*vc[k] as isize + off[k]) as usize; }
            let ef = flat3(&es, &dstride);
            adj[ef] |= EDGE_BOUNDARY;
            let e = emap.entry(key).or_insert([0u64; N_EDGE_FEATURES]);
            e[0] += 1;
            e[1] += dist;
            if dist > e[2] { e[2] = dist; }
        }
    }

    let mut keys: Vec<(i64, i64)> = emap.keys().copied().collect();
    keys.sort_unstable();
    let n_edges = keys.len();
    let mut edge_index = vec![0i64; 2 * n_edges];
    let mut edge_feats = vec![0u64; N_EDGE_FEATURES * n_edges];
    for (i, k) in keys.iter().enumerate() {
        edge_index[i] = k.0;
        edge_index[n_edges + i] = k.1;
        let e = emap[k];
        for j in 0..N_EDGE_FEATURES { edge_feats[i * N_EDGE_FEATURES + j] = e[j]; }
    }

    Out { adj, ddims, node_feats, n_reg, nf, labels, edge_index, edge_feats, n_edges }
}

#[pyfunction]
#[pyo3(signature = (data, mask, band_of, deleted = None, iterator_divisor = None))]
fn band_build<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'_, u8>,
    mask: PyReadonlyArrayDyn<'_, u8>,
    band_of: PyReadonlyArray1<'_, u8>,
    deleted: Option<PyReadonlyArray1<'_, u8>>,
    iterator_divisor: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray3<i64>>,
    Bound<'py, PyArray3<u8>>,
)> {
    let dshape = data.shape();
    if dshape.len() != 4 { return Err(PyValueError::new_err("data must be (D,H,W,C)")); }
    let dims = [dshape[0], dshape[1], dshape[2]];
    let c = dshape[3];
    if dims[0] < 2 || dims[1] < 2 || dims[2] < 2 {
        return Err(PyValueError::new_err("each spatial dim must be >= 2"));
    }
    let mshape = mask.shape();
    if mshape != [dims[0], dims[1], dims[2]] {
        return Err(PyValueError::new_err("mask must be (D,H,W) matching data spatial dims"));
    }
    if band_of.len() != 256 {
        return Err(PyValueError::new_err("band_of must be length 256"));
    }

    // deleted mask: default to all-zero (no deletion) if not supplied.
    let deleted_owned: Vec<u8> = match &deleted {
        Some(arr) => {
            if arr.len() != 256 {
                return Err(PyValueError::new_err("deleted must be length 256"));
            }
            arr.as_slice()?.to_vec()
        }
        None => vec![0u8; 256],
    };

    let divisor = iterator_divisor.unwrap_or(4.12).max(1.0);

    let data_s = data.as_slice()?;
    let mask_s = mask.as_slice()?;
    let band_s = band_of.as_slice()?;

    let out = py.allow_threads(|| process(data_s, mask_s, band_s, &deleted_owned, dims, c, divisor));

    let feat = Array2::from_shape_vec((out.n_reg, out.nf), out.node_feats).unwrap();
    let ei = Array2::from_shape_vec((2, out.n_edges), out.edge_index).unwrap();
    let ef = Array2::from_shape_vec((out.n_edges, N_EDGE_FEATURES), out.edge_feats).unwrap();
    let lab = Array3::from_shape_vec((dims[0], dims[1], dims[2]), out.labels).unwrap();
    let adj = Array3::from_shape_vec((out.ddims[0], out.ddims[1], out.ddims[2]), out.adj).unwrap();

    Ok((
        PyArray2::from_owned_array_bound(py, feat),
        PyArray2::from_owned_array_bound(py, ei),
        PyArray2::from_owned_array_bound(py, ef),
        PyArray3::from_owned_array_bound(py, lab),
        PyArray3::from_owned_array_bound(py, adj),
    ))
}

#[pymodule]
fn fastloops_band(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(band_build, m)?)?;
    Ok(())
}