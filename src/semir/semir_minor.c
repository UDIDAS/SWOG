/*
 * SEMIR graph minor construction in C.
 *
 * Implements the expanded-tensor flood-fill contraction from:
 *   "SEMIR: Semantic Minor-Induced Representation Learning
 *    on Graphs for Visual Segmentation" (ICML 2026)
 *
 * Called from Python via ctypes.
 *
 * The key insight vs naive connected components:
 *   Each supernode has a SEED voxel. When expanding via flood-fill,
 *   every candidate is compared against the SEED intensity, NOT its
 *   immediate neighbor. This prevents transitive intensity drift.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---------- Bitflags for expanded tensor entries ---------- */
#define FLAG_VISITED   0x01
#define FLAG_DELETED   0x02
#define FLAG_BOUNDARY  0x04
#define FLAG_EDGE_DEL  0x08

/* ---------- Stack for flood fill ---------- */
typedef struct {
    int32_t *data;
    int      top;
    int      capacity;
} Stack;

static void stack_init(Stack *s, int cap) {
    s->data = (int32_t *)malloc(cap * 3 * sizeof(int32_t));
    s->top = 0;
    s->capacity = cap;
}

static void stack_push(Stack *s, int y, int x, int z) {
    if (s->top >= s->capacity) {
        s->capacity *= 2;
        s->data = (int32_t *)realloc(s->data, s->capacity * 3 * sizeof(int32_t));
    }
    int i = s->top * 3;
    s->data[i]     = y;
    s->data[i + 1] = x;
    s->data[i + 2] = z;
    s->top++;
}

static int stack_pop(Stack *s, int *y, int *x, int *z) {
    if (s->top <= 0) return 0;
    s->top--;
    int i = s->top * 3;
    *y = s->data[i];
    *x = s->data[i + 1];
    *z = s->data[i + 2];
    return 1;
}

static void stack_free(Stack *s) {
    free(s->data);
    s->data = NULL;
}

/* ---------- 6-connected neighbor offsets (in voxel space) ---------- */
static const int NEIGH_6[6][3] = {
    {-1,  0,  0}, { 1,  0,  0},
    { 0, -1,  0}, { 0,  1,  0},
    { 0,  0, -1}, { 0,  0,  1}
};

/* ---------- GCD for coprime traversal ---------- */
static int gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a < 0 ? -a : a;
}

static int find_coprime(int n) {
    if (n <= 1) return 1;
    /* Find a step size coprime to n, starting near n/sqrt(5) (golden ratio) */
    int s = (int)(n / 2.236);
    if (s < 1) s = 1;
    for (int i = s; i < n; i++) {
        if (gcd(n, i) == 1) return i;
    }
    for (int i = s - 1; i >= 1; i--) {
        if (gcd(n, i) == 1) return i;
    }
    return 1;
}

/* ---------- Main construction ----------
 *
 * Parameters:
 *   volume    : float64 array [H][W][D], normalized to [0,1]
 *   H, W, D   : dimensions
 *   psi       : contraction threshold (merge if |I_p - I_seed| <= psi)
 *   alpha     : edge deletion threshold
 *   beta_min  : minimum supernode size
 *   beta_max  : maximum supernode size
 *   m_min     : minimum mean intensity for supernode
 *   m_max     : maximum mean intensity for supernode
 *   labels_out: int32 array [H][W][D], filled with supernode IDs (1-based, 0 = deleted)
 *
 * Returns: number of supernodes after deletion
 */
int semir_minor_construct(
    const double *volume,
    int H, int W, int D,
    double psi, double alpha,
    int beta_min, int beta_max,
    double m_min, double m_max,
    int32_t *labels_out)
{
    int HWD = H * W * D;

    /* Expanded tensor dimensions */
    int TH = 2 * H - 1;
    int TW = 2 * W - 1;
    int TD = 2 * D - 1;
    long T_size = (long)TH * TW * TD;

    /* Allocate expanded tensor (single byte per entry) */
    uint8_t *T = (uint8_t *)calloc(T_size, 1);
    if (!T) return -1;

    /* Supernode labels for each voxel */
    memset(labels_out, 0, HWD * sizeof(int32_t));

    /* Per-supernode storage: dynamically grown */
    int sn_cap = 4096;
    int sn_count = 0;
    int    *sn_area     = (int *)   malloc(sn_cap * sizeof(int));
    double *sn_sum_int  = (double *)malloc(sn_cap * sizeof(double));

    /* Coprime steps for pseudo-random traversal */
    int sh = find_coprime(H);
    int sw = find_coprime(W);
    int sd = find_coprime(D);

    /* Random-ish starting offsets */
    int r0 = H / 3, c0 = W / 5, l0 = D / 7;

    /* Flood-fill stack */
    Stack stk;
    stack_init(&stk, 65536);

    /* Macro for tensor indexing */
    #define T_IDX(y, x, z) ((long)(y) * TW * TD + (long)(x) * TD + (long)(z))
    #define V_IDX(r, c, l) ((r) * W * D + (c) * D + (l))

    /* Phase 1: Edge contraction via flood-fill */
    for (int i = 0; i < H; i++) {
        int r = (r0 + (long)i * sh) % H;
        for (int j = 0; j < W; j++) {
            int c = (c0 + (long)j * sw) % W;
            for (int k = 0; k < D; k++) {
                int l = (l0 + (long)k * sd) % D;

                /* Tensor position for this voxel */
                int ty = 2 * r, tx = 2 * c, tz = 2 * l;

                /* Skip if already visited */
                if (T[T_IDX(ty, tx, tz)] & FLAG_VISITED)
                    continue;

                /* Seed intensity (canonical voxel) */
                double v_seed = volume[V_IDX(r, c, l)];

                /* Start flood fill */
                stk.top = 0;
                stack_push(&stk, ty, tx, tz);

                int area = 0;
                double sum_int = 0.0;

                while (stk.top > 0) {
                    int py, px, pz;
                    stack_pop(&stk, &py, &px, &pz);

                    /* Check visited */
                    long tidx = T_IDX(py, px, pz);
                    if (T[tidx] & FLAG_VISITED)
                        continue;
                    T[tidx] |= FLAG_VISITED;

                    /* Voxel coordinates */
                    int vr = py / 2, vc = px / 2, vl = pz / 2;
                    int vidx = V_IDX(vr, vc, vl);

                    /* Accumulate */
                    area++;
                    sum_int += volume[vidx];

                    /* Assign tentative label (1-based) */
                    labels_out[vidx] = sn_count + 1;

                    /* Explore 6-connected neighbors */
                    for (int n = 0; n < 6; n++) {
                        int nr = vr + NEIGH_6[n][0];
                        int nc = vc + NEIGH_6[n][1];
                        int nl = vl + NEIGH_6[n][2];

                        /* Bounds check */
                        if (nr < 0 || nr >= H ||
                            nc < 0 || nc >= W ||
                            nl < 0 || nl >= D)
                            continue;

                        int ny = 2 * nr, nx = 2 * nc, nz = 2 * nl;

                        /* Skip if neighbor already visited */
                        if (T[T_IDX(ny, nx, nz)] & FLAG_VISITED)
                            continue;

                        /* Edge position in expanded tensor */
                        int ey = py + NEIGH_6[n][0];
                        int ex = px + NEIGH_6[n][1];
                        int ez = pz + NEIGH_6[n][2];

                        /* Skip if edge already processed */
                        long eidx = T_IDX(ey, ex, ez);
                        if (T[eidx] & (FLAG_VISITED | FLAG_EDGE_DEL))
                            continue;

                        /* Intensity difference vs SEED (not vs current!) */
                        double diff = fabs(volume[V_IDX(nr, nc, nl)] - v_seed);

                        if (diff <= psi) {
                            /* Merge: mark edge as contracted */
                            T[eidx] |= FLAG_VISITED;
                            stack_push(&stk, ny, nx, nz);
                        } else if (diff >= alpha) {
                            /* Paper Algorithm 3: mark edge as deleted during
                               flood-fill. Strong boundary detected. */
                            T[eidx] |= FLAG_EDGE_DEL;
                            T[T_IDX(py, px, pz)] |= FLAG_BOUNDARY;
                        }
                    }
                }

                /* Supernode complete — check deletion criteria */
                double mean_int = (area > 0) ? sum_int / area : 0.0;

                int keep = (area >= beta_min && area <= beta_max &&
                            mean_int >= m_min && mean_int <= m_max);

                if (keep) {
                    /* Store supernode stats */
                    if (sn_count >= sn_cap) {
                        sn_cap *= 2;
                        sn_area    = (int *)   realloc(sn_area,    sn_cap * sizeof(int));
                        sn_sum_int = (double *)realloc(sn_sum_int, sn_cap * sizeof(double));
                    }
                    sn_area[sn_count]    = area;
                    sn_sum_int[sn_count] = sum_int;
                    sn_count++;
                } else {
                    /* Delete: clear labels for this supernode's voxels.
                       We can't easily un-label them here without storing
                       the voxel list, so we mark with label 0 in a second pass.
                       For now, record deletion: label = -(sn_count+1) */
                    /* Actually, just use a post-pass. Mark as negative. */
                    int del_label = sn_count + 1;
                    /* We'll handle deletion in a post-pass by checking
                       if the supernode was kept */
                    sn_count++;
                    if (sn_count > sn_cap) {
                        sn_cap = sn_count * 2;
                        sn_area    = (int *)   realloc(sn_area,    sn_cap * sizeof(int));
                        sn_sum_int = (double *)realloc(sn_sum_int, sn_cap * sizeof(double));
                    }
                    sn_area[sn_count - 1]    = -1;  /* mark as deleted */
                    sn_sum_int[sn_count - 1] = 0.0;
                }
            }
        }
    }

    /* Phase 2: Relabel — compact surviving supernodes, zero out deleted ones */
    int *remap = (int *)calloc(sn_count + 1, sizeof(int));
    int new_id = 0;
    for (int s = 0; s < sn_count; s++) {
        if (sn_area[s] >= 0) {
            new_id++;
            remap[s + 1] = new_id;  /* old 1-based -> new 1-based */
        } else {
            remap[s + 1] = 0;  /* deleted */
        }
    }

    /* Apply remap to labels */
    for (int i = 0; i < HWD; i++) {
        int old = labels_out[i];
        if (old > 0 && old <= sn_count) {
            labels_out[i] = remap[old];
        } else {
            labels_out[i] = 0;
        }
    }

    /* Cleanup */
    stack_free(&stk);
    free(T);
    free(sn_area);
    free(sn_sum_int);
    free(remap);

    #undef T_IDX
    #undef V_IDX

    return new_id;
}
