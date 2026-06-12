//! SEMIR graph minor construction in Rust.
//!
//! Implements the expanded binary tensor flood-fill from:
//!   "SEMIR: Semantic Minor-Induced Representation Learning
//!    on Graphs for Visual Segmentation" (ICML 2026)
//!
//! Key design: each voxel and each edge between adjacent voxels occupies
//! a position in the expanded tensor T of size (2H-1)×(2W-1)×(2D-1).
//! Voxels are at even coordinates, edges at odd.
//!
//! The flood-fill compares each candidate neighbor against the SEED
//! voxel's intensity (canonical anchoring), preventing transitive drift.
//! Every edge is marked exactly once — either contracted or boundary.

use std::collections::VecDeque;

// --- Bitflags for expanded tensor entries ---
const VISITED: u8 = 0x01;
const EDGE_DEL: u8 = 0x02;
const BOUNDARY: u8 = 0x04;

// --- 6-connected neighbor offsets ---
const NEIGH6: [(i32, i32, i32); 6] = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
];

/// Find a coprime step for pseudo-random traversal.
fn coprime_step(n: usize) -> usize {
    if n <= 1 { return 1; }
    let start = (n as f64 / 2.236) as usize;
    let start = start.max(1);
    // Search upward
    for s in start..n {
        if gcd(n, s) == 1 { return s; }
    }
    // Search downward
    for s in (1..start).rev() {
        if gcd(n, s) == 1 { return s; }
    }
    1
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Expanded tensor: interleaves voxel nodes (even coords) and edges (odd coords).
struct ExpandedTensor {
    data: Vec<u8>,
    tw: usize,
    td: usize,
}

impl ExpandedTensor {
    fn new(h: usize, w: usize, d: usize) -> Self {
        let th = 2 * h - 1;
        let tw = 2 * w - 1;
        let td = 2 * d - 1;
        Self {
            data: vec![0u8; th * tw * td],
            tw,
            td,
        }
    }

    #[inline(always)]
    fn idx(&self, y: usize, x: usize, z: usize) -> usize {
        y * self.tw * self.td + x * self.td + z
    }

    #[inline(always)]
    fn get(&self, y: usize, x: usize, z: usize) -> u8 {
        self.data[self.idx(y, x, z)]
    }

    #[inline(always)]
    fn set_flag(&mut self, y: usize, x: usize, z: usize, flag: u8) {
        let i = self.idx(y, x, z);
        self.data[i] |= flag;
    }

    #[inline(always)]
    fn has_flag(&self, y: usize, x: usize, z: usize, flag: u8) -> bool {
        self.data[self.idx(y, x, z)] & flag != 0
    }
}

/// Per-supernode stats accumulated during flood-fill.
struct SupernodeStats {
    area: u64,
    intensity_sum: f64,
    deleted: bool,
}

/// Main entry point — called from Python via ctypes.
///
/// # Safety
/// Caller must provide valid pointers to arrays of the correct sizes.
#[no_mangle]
pub unsafe extern "C" fn semir_minor_construct(
    volume: *const f64,
    h: i32, w: i32, d: i32,
    psi: f64, alpha: f64,
    beta_min: i32, beta_max: i32,
    m_min: f64, m_max: f64,
    labels_out: *mut i32,
) -> i32 {
    let h = h as usize;
    let w = w as usize;
    let d = d as usize;
    let hwd = h * w * d;
    let beta_min = beta_min as u64;
    let beta_max = beta_max as u64;

    // Wrap raw pointers in slices
    let vol = std::slice::from_raw_parts(volume, hwd);
    let labels = std::slice::from_raw_parts_mut(labels_out, hwd);

    // Zero labels
    labels.iter_mut().for_each(|l| *l = 0);

    // Expanded tensor
    let mut tensor = ExpandedTensor::new(h, w, d);

    // Supernode storage
    let mut supernodes: Vec<SupernodeStats> = Vec::with_capacity(4096);

    // Coprime traversal (pseudo-random seed order, avoids directional bias)
    let sh = coprime_step(h);
    let sw = coprime_step(w);
    let sd = coprime_step(d);
    let r0 = h / 3;
    let c0 = w / 5;
    let l0 = d / 7;

    // BFS queue for flood-fill (BFS gives more uniform region shapes than DFS)
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::with_capacity(65536);

    // Volume indexing helper
    let vidx = |r: usize, c: usize, l: usize| -> usize { r * w * d + c * d + l };

    // --- Phase 1: Flood-fill contraction + inline node deletion ---
    for i in 0..h {
        let r = (r0 + i * sh) % h;
        for j in 0..w {
            let c = (c0 + j * sw) % w;
            for k in 0..d {
                let l = (l0 + k * sd) % d;

                let ty = 2 * r;
                let tx = 2 * c;
                let tz = 2 * l;

                // Skip if already visited (belongs to another supernode)
                if tensor.has_flag(ty, tx, tz, VISITED) {
                    continue;
                }

                // --- Start new supernode from this seed ---
                let seed_intensity = vol[vidx(r, c, l)];
                let sn_id = supernodes.len(); // 0-based

                let mut area: u64 = 0;
                let mut intensity_sum: f64 = 0.0;

                queue.clear();
                queue.push_back((ty, tx, tz));

                while let Some((py, px, pz)) = queue.pop_front() {
                    // Already visited? (could be pushed multiple times)
                    if tensor.has_flag(py, px, pz, VISITED) {
                        continue;
                    }

                    // Size cap: stop growing at beta_max.
                    // Don't mark as visited — a later seed will claim this voxel.
                    if area >= beta_max {
                        continue;
                    }

                    // Mark visited and assign to this supernode
                    tensor.set_flag(py, px, pz, VISITED);
                    let vr = py / 2;
                    let vc = px / 2;
                    let vl = pz / 2;
                    let vi = vidx(vr, vc, vl);

                    area += 1;
                    intensity_sum += vol[vi];
                    // 1-based label (sn_id 0 → label 1)
                    labels[vi] = (sn_id + 1) as i32;

                    // Explore 6-connected neighbors
                    for &(dr, dc, dl) in &NEIGH6 {
                        let nr = vr as i32 + dr;
                        let nc = vc as i32 + dc;
                        let nl = vl as i32 + dl;

                        // Bounds check
                        if nr < 0 || nr >= h as i32 ||
                           nc < 0 || nc >= w as i32 ||
                           nl < 0 || nl >= d as i32 {
                            continue;
                        }
                        let nr = nr as usize;
                        let nc = nc as usize;
                        let nl = nl as usize;

                        let ny = 2 * nr;
                        let nx = 2 * nc;
                        let nz = 2 * nl;

                        // Skip if neighbor already visited
                        if tensor.has_flag(ny, nx, nz, VISITED) {
                            continue;
                        }

                        // Edge position in expanded tensor
                        let ey = (py as i32 + dr) as usize;
                        let ex = (px as i32 + dc) as usize;
                        let ez = (pz as i32 + dl) as usize;

                        // Skip if edge already decided
                        if tensor.get(ey, ex, ez) & (VISITED | EDGE_DEL) != 0 {
                            continue;
                        }

                        // Intensity diff vs SEED (canonical anchoring)
                        let diff = (vol[vidx(nr, nc, nl)] - seed_intensity).abs();

                        if diff <= psi {
                            // Strong merge: within contraction threshold
                            tensor.set_flag(ey, ex, ez, VISITED);
                            queue.push_back((ny, nx, nz));
                        } else if diff > alpha {
                            // Hard boundary: delete edge
                            tensor.set_flag(ey, ex, ez, EDGE_DEL);
                            tensor.set_flag(py, px, pz, BOUNDARY);
                        } else {
                            // psi < diff <= alpha: merge but mark boundary.
                            // This is the key zone — the seed can still claim
                            // these voxels but they're at the edge of the region.
                            tensor.set_flag(ey, ex, ez, VISITED);
                            tensor.set_flag(py, px, pz, BOUNDARY);
                            queue.push_back((ny, nx, nz));
                        }
                    }
                }

                // Supernode complete — check deletion criteria
                let mean_int = if area > 0 {
                    intensity_sum / area as f64
                } else {
                    0.0
                };

                let keep = area >= beta_min
                    && area <= beta_max
                    && mean_int >= m_min
                    && mean_int <= m_max;

                supernodes.push(SupernodeStats {
                    area,
                    intensity_sum,
                    deleted: !keep,
                });
            }
        }
    }

    // --- Phase 2: Iterative supernode merging ---
    // "Contraction recurses until no further valid merges exist" (paper Sec 3.2)
    //
    // After the initial flood-fill, adjacent supernodes whose canonical (seed)
    // intensities differ by ≤ psi are merged. Repeat until convergence.
    // This is what reduces ~100K supernodes down to ~1K.
    //
    // We use Union-Find for efficient merging.

    let n_sn = supernodes.len();

    // Union-Find
    let mut parent: Vec<usize> = (0..n_sn + 1).collect(); // 1-based IDs
    let mut uf_size: Vec<u64> = vec![0; n_sn + 1];
    let mut uf_intensity_sum: Vec<f64> = vec![0.0; n_sn + 1];
    let mut uf_seed_intensity: Vec<f64> = vec![0.0; n_sn + 1];

    // Initialize from supernodes
    for (s, sn) in supernodes.iter().enumerate() {
        let id = s + 1; // 1-based
        uf_size[id] = sn.area;
        uf_intensity_sum[id] = sn.intensity_sum;
        // Canonical intensity = mean intensity of the supernode
        uf_seed_intensity[id] = if sn.area > 0 {
            sn.intensity_sum / sn.area as f64
        } else {
            0.0
        };
    }

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            parent[r] = parent[parent[r]]; // path compression
            r = parent[r];
        }
        r
    }

    // Iterative merge: find adjacent supernode pairs and merge if similar
    let mut changed = true;
    let mut round = 0;
    while changed {
        changed = false;
        round += 1;

        // Scan all adjacent voxel pairs to find supernode boundary pairs
        for axis in 0..3usize {
            let (dh, dw, dd): (usize, usize, usize) = match axis {
                0 => (1, 0, 0),
                1 => (0, 1, 0),
                _ => (0, 0, 1),
            };

            let h_end = if axis == 0 { h - 1 } else { h };
            let w_end = if axis == 1 { w - 1 } else { w };
            let d_end = if axis == 2 { d - 1 } else { d };

            for r in 0..h_end {
                for c in 0..w_end {
                    for l in 0..d_end {
                        let a = labels[vidx(r, c, l)];
                        let b = labels[vidx(r + dh, c + dw, l + dd)];

                        if a <= 0 || b <= 0 || a == b {
                            continue;
                        }

                        let ra = find(&mut parent, a as usize);
                        let rb = find(&mut parent, b as usize);
                        if ra == rb {
                            continue; // already merged
                        }

                        // Check if canonical intensities are within alpha
                        // (psi for voxel-level contraction, alpha for supernode-level merge)
                        let diff = (uf_seed_intensity[ra] - uf_seed_intensity[rb]).abs();
                        if diff <= alpha {
                            // Check combined size doesn't exceed beta_max
                            let combined = uf_size[ra] + uf_size[rb];
                            if combined <= beta_max {
                                // Merge: smaller into larger
                                let (big, small) = if uf_size[ra] >= uf_size[rb] {
                                    (ra, rb)
                                } else {
                                    (rb, ra)
                                };
                                parent[small] = big;
                                uf_size[big] = combined;
                                uf_intensity_sum[big] += uf_intensity_sum[small];
                                // Update canonical intensity to weighted mean
                                uf_seed_intensity[big] = uf_intensity_sum[big] / combined as f64;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Safety: prevent infinite loops
        if round > 100 {
            break;
        }
    }

    // --- Phase 3: Relabel with merged IDs + node deletion ---
    // Build final remap: find root for each supernode, apply deletion criteria
    let mut root_to_new: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    let mut new_id: i32 = 0;

    for s in 1..=n_sn {
        let root = find(&mut parent, s);

        if root_to_new.contains_key(&root) {
            continue;
        }

        let area = uf_size[root];
        let mean_int = if area > 0 {
            uf_intensity_sum[root] / area as f64
        } else {
            0.0
        };

        let keep = area >= beta_min
            && area <= beta_max
            && mean_int >= m_min
            && mean_int <= m_max;

        if keep {
            new_id += 1;
            root_to_new.insert(root, new_id);
        } else {
            root_to_new.insert(root, 0);
        }
    }

    // Apply remap
    for l in labels.iter_mut() {
        let old = *l;
        if old > 0 && (old as usize) <= n_sn {
            let root = find(&mut parent, old as usize);
            *l = *root_to_new.get(&root).unwrap_or(&0);
        } else {
            *l = 0;
        }
    }

    new_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_volume() {
        let h = 10usize;
        let w = 10;
        let d = 10;
        let vol = vec![0.5f64; h * w * d];
        let mut labels = vec![0i32; h * w * d];

        let n = unsafe {
            semir_minor_construct(
                vol.as_ptr(), h as i32, w as i32, d as i32,
                0.03, 0.20, 1, 500000, 0.0, 1.0,
                labels.as_mut_ptr(),
            )
        };

        assert_eq!(n, 1, "Uniform volume should produce 1 supernode");
        assert!(labels.iter().all(|&l| l == 1), "All voxels should be label 1");
    }

    #[test]
    fn test_two_regions() {
        // 10x10x10 volume: left half = 0.2, right half = 0.8
        let h = 10usize;
        let w = 10;
        let d = 10;
        let mut vol = vec![0.0f64; h * w * d];
        for r in 0..h {
            for c in 0..w {
                for l in 0..d {
                    vol[r * w * d + c * d + l] = if l < 5 { 0.2 } else { 0.8 };
                }
            }
        }
        let mut labels = vec![0i32; h * w * d];

        let n = unsafe {
            semir_minor_construct(
                vol.as_ptr(), h as i32, w as i32, d as i32,
                0.05, 0.20, 1, 500000, 0.0, 1.0,
                labels.as_mut_ptr(),
            )
        };

        // Should produce exactly 2 supernodes (diff=0.6 > alpha=0.2)
        assert_eq!(n, 2, "Two-region volume should produce 2 supernodes");
        // All labels should be non-zero
        assert!(labels.iter().all(|&l| l > 0), "No deleted voxels");
    }

    #[test]
    fn test_size_cap() {
        // Uniform volume with small beta_max
        let h = 20usize;
        let w = 20;
        let d = 20;
        let vol = vec![0.5f64; h * w * d];
        let mut labels = vec![0i32; h * w * d];

        let n = unsafe {
            semir_minor_construct(
                vol.as_ptr(), h as i32, w as i32, d as i32,
                0.03, 0.20, 1, 1000, // cap at 1000 voxels
                0.0, 1.0,
                labels.as_mut_ptr(),
            )
        };

        // 8000 voxels / 1000 cap → multiple supernodes (exact count depends on traversal order)
        assert!(n >= 4 && n <= 50, "Should produce multiple supernodes, got {}", n);
        assert!(labels.iter().all(|&l| l > 0), "No deleted voxels with beta_min=1");
    }

    #[test]
    fn test_gradient() {
        // Gradient along z: 0.0 to 1.0
        let h = 4usize;
        let w = 4;
        let d = 8;
        let mut vol = vec![0.0f64; h * w * d];
        for r in 0..h {
            for c in 0..w {
                for l in 0..d {
                    vol[r * w * d + c * d + l] = l as f64 / 7.0;
                }
            }
        }
        let mut labels = vec![0i32; h * w * d];

        let n = unsafe {
            semir_minor_construct(
                vol.as_ptr(), h as i32, w as i32, d as i32,
                0.03, 0.20, 1, 500000, 0.0, 1.0,
                labels.as_mut_ptr(),
            )
        };

        // With alpha=0.20, seed at 0.0 covers [0, 0.20], seed at ~0.28 covers [0.08, 0.48], etc.
        // Should produce ~3-5 supernodes
        assert!(n >= 2 && n <= 8, "Gradient should produce 2-8 supernodes, got {}", n);
        assert!(labels.iter().all(|&l| l > 0), "No deleted voxels");
    }
}
