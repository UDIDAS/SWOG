// fastloops_search: the LEAN Stage-1 intensity-band search flood.
//
// NOT merge_and_cut. Computes the ONE thing the sweep oracle needs: per emitted region,
// (#fg, #bg) voxel counts under the mask. No moments, edges, canonical voxel, doubled adj,
// labels, edge_index, or deletion. Cheap and fast so the sweep can flood a volume hundreds
// of times while searching for the intensity-merge dictionary.
//
// CONTRACT (channel-last, C-contiguous, u8). 3D only. The merge predicate keys on channel 0
// (grayscale CT, C=1).
//   data : (D, H, W, C) u8   -- the volume
//   mask : (D, H, W)   u8     -- ground-truth foreground, {0,1}
//   bands: (256*256)   u8     -- row-major band table; bands[seed*256 + neighbor] != 0 iff
//                                neighbor's intensity is in seed's merge list. Reflexive/
//                                symmetric by construction. All-zero row (not even self) ->
//                                that intensity seeds singletons.
//
// RETURNS (fg_counts, bg_counts), each (R,) u64, one per flooded region in seed order.
// Driver computes IoU = sum_{fg>bg} fg / (sum_{fg>bg} bg + GT).

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

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

fn search_flood(
    data: &[u8], mask: &[u8], bands: &[u8],
    dims: [usize; 3], c: usize, divisor: f64,
) -> (Vec<u64>, Vec<u64>) {
    let n_voxels = dims[0] * dims[1] * dims[2];
    let mut vstride = [0usize; 3];
    vstride[2] = 1; vstride[1] = dims[2]; vstride[0] = dims[1] * dims[2];
    let mut step = [0usize; 3];
    let mut start = [0usize; 3];
    for k in 0..3 { step[k] = coprime_step(dims[k], divisor); start[k] = step[k] % dims[k]; }
    const OFFS: [[isize; 3]; 6] = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
    let mut visited = vec![false; n_voxels];
    let mut stack: Vec<usize> = Vec::new();
    let mut fg_counts: Vec<u64> = Vec::new();
    let mut bg_counts: Vec<u64> = Vec::new();
    for idx in 0..n_voxels {
        let mut rem = idx;
        let mut vseed = [0usize; 3];
        for k in (0..3).rev() { let i_k = rem % dims[k]; rem /= dims[k]; vseed[k] = (start[k] + i_k * step[k]) % dims[k]; }
        let seed_flat = vseed[0]*vstride[0] + vseed[1]*vstride[1] + vseed[2]*vstride[2];
        if visited[seed_flat] { continue; }
        let seed_i = data[seed_flat * c] as usize;
        let row = seed_i * 256;
        let mut fg: u64 = 0; let mut bg: u64 = 0;
        visited[seed_flat] = true; stack.clear(); stack.push(seed_flat);
        while let Some(vf) = stack.pop() {
            if mask[vf] != 0 { fg += 1; } else { bg += 1; }
            let z = vf / vstride[0]; let r = vf % vstride[0];
            let y = r / vstride[1]; let x = r % vstride[1];
            for off in &OFFS {
                let nz = z as isize + off[0]; let ny = y as isize + off[1]; let nx = x as isize + off[2];
                if nz < 0 || nz >= dims[0] as isize || ny < 0 || ny >= dims[1] as isize || nx < 0 || nx >= dims[2] as isize { continue; }
                let nf = nz as usize*vstride[0] + ny as usize*vstride[1] + nx as usize;
                if visited[nf] { continue; }
                let ni = data[nf * c] as usize;
                if bands[row + ni] != 0 { visited[nf] = true; stack.push(nf); }
            }
        }
        fg_counts.push(fg); bg_counts.push(bg);
    }
    (fg_counts, bg_counts)
}

#[pyfunction]
#[pyo3(signature = (data, mask, bands, divisor = 2.0))]
fn intensity_search<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'_, u8>,
    mask: PyReadonlyArrayDyn<'_, u8>,
    bands: PyReadonlyArray1<'_, u8>,
    divisor: f64,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let dshape = data.shape();
    if dshape.len() != 4 { return Err(pyo3::exceptions::PyValueError::new_err("data must be (D,H,W,C)")); }
    let dims = [dshape[0], dshape[1], dshape[2]];
    let c = dshape[3];
    let mshape = mask.shape();
    if mshape != [dims[0], dims[1], dims[2]] {
        return Err(pyo3::exceptions::PyValueError::new_err("mask must be (D,H,W) matching data spatial dims"));
    }
    if bands.len() != 256 * 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("bands must be length 256*256"));
    }
    let data_s = data.as_slice()?;
    let mask_s = mask.as_slice()?;
    let bands_s = bands.as_slice()?;
    let (fg, bg) = py.allow_threads(|| search_flood(data_s, mask_s, bands_s, dims, c, divisor));
    Ok((
        PyArray1::from_owned_array_bound(py, Array1::from(fg)),
        PyArray1::from_owned_array_bound(py, Array1::from(bg)),
    ))
}

#[pymodule]
fn fastloops_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(intensity_search, m)?)?;
    Ok(())
}