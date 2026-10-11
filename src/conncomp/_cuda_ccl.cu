// Block-based union-find connected-component labelling on the six cube-face grids (CUDA).
//
// Layout: for each form b and face f (fb = 6 b + f) an M x M grid; node index ((fb * M) + i) * M + j.
// Connectivity: 4-neighbours for both sign classes, diagonals only for the class `eight` (8/4 digital topology).
// Points on cube edges/corners are duplicated on 2/3 faces; `seams` lists (node, canonical node) pairs per form.
//
// Kernels:
//   local_tiles   T x T tiles labelled in shared memory (union-find with atomicMin and path splitting), roots written
//                 to global memory
//   border_merge  global union-find across tile borders (each grid edge is handled by exactly one of the kernels)
//   merge_seams   union of the duplicated cube-edge points
//   compress      parent[x] = root(x) with read-only root finding (no halving writes: they would race)
//   form_trees    region trees from the labels, one block per form (see below)
//   fused_trees   coefficients -> signs -> union-find over row runs -> region trees, one block per form (see below)

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int T = 16;  // tile size (T x T threads per block)

__device__ __forceinline__ int find_shared(volatile int *lab, int x) {
    int p = lab[x];
    while (p != x) {
        x = p;
        p = lab[x];
    }
    return x;
}

// Root finding with path splitting (every visited node -> its grandparent), for the union phase in shared memory.
// The grandparent is an ancestor, so connectivity is preserved; the pointer is replaced by atomicCAS only if it is
// unchanged, so a concurrent link made by atomicMin in union_shared is never overwritten.
__device__ __forceinline__ int find_splitting(int *lab, int x) {
    volatile int *v = lab;  // plain reads see either the old parent or a later ancestor
    int p = v[x];
    while (p != x) {
        const int g = v[p];
        if (g != p) atomicCAS(&lab[x], p, g);
        x = p;
        p = g;
    }
    return x;
}

__device__ __forceinline__ void union_shared(int *lab, int a, int b) {
    while (true) {
        a = find_splitting(lab, a);
        b = find_splitting(lab, b);
        if (a == b) return;
        if (a < b) {
            const int t = a;
            a = b;
            b = t;
        }
        const int old = atomicMin(&lab[a], b);  // link the larger root under the smaller
        if (old == a) return;
        a = old;  // a was no longer a root: continue with its new parent
    }
}

__device__ __forceinline__ int find_global(const int *parent, int x) {
    const volatile int *p = parent;
    int q = p[x];
    while (q != x) {
        x = q;
        q = p[x];
    }
    return x;
}

__device__ __forceinline__ void union_global(int *parent, int a, int b) {
    while (true) {
        a = find_global(parent, a);
        b = find_global(parent, b);
        if (a == b) return;
        if (a < b) {
            const int t = a;
            a = b;
            b = t;
        }
        const int old = atomicMin(&parent[a], b);
        if (old == a) return;
        a = old;
    }
}

__global__ void local_tiles(const uint8_t *signs, int *parent, int M, int eight) {
    __shared__ int lab[T * T];
    __shared__ uint8_t sg[T * T];
    const int fb = blockIdx.z;
    const int x = threadIdx.x, y = threadIdx.y;
    const int i = blockIdx.y * T + y, j = blockIdx.x * T + x;
    const int l = y * T + x;
    const bool in = i < M && j < M;
    const long long base = (long long)fb * M * M;
    sg[l] = in ? signs[base + (long long)i * M + j] : 255;
    lab[l] = l;
    __syncthreads();
    const uint8_t s = sg[l];
    if (in) {
        if (x > 0 && sg[l - 1] == s) union_shared(lab, l, l - 1);
        if (y > 0 && sg[l - T] == s) union_shared(lab, l, l - T);
        if (s == eight) {
            if (x > 0 && y > 0 && sg[l - T - 1] == s) union_shared(lab, l, l - T - 1);
            if (x < T - 1 && y > 0 && sg[l - T + 1] == s) union_shared(lab, l, l - T + 1);
        }
    }
    __syncthreads();
    if (in) {
        const int r = find_shared(lab, l);
        const int ri = blockIdx.y * T + r / T, rj = blockIdx.x * T + r % T;
        parent[base + (long long)i * M + j] = (int)(base + (long long)ri * M + rj);
    }
}

// One thread per tile-border node (list of in-face offsets i * M + j, shared by all faces and forms).
__global__ void border_merge(const uint8_t *signs, int *parent, const int *border, int nb, long long n_faces, int M,
                             int eight) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_faces * nb) return;
    const long long fb = k / nb;
    const int q = border[k % nb], i = q / M, j = q % M;
    const bool last_col = (j % T) == T - 1, last_row = (i % T) == T - 1, first_col = (j % T) == 0;
    const int p = (int)(fb * M * M + q);
    const uint8_t s = signs[p];
    // edges leaving the tile: right, down, and (for the class `eight`) down-right, down-left
    if (last_col && j + 1 < M && signs[p + 1] == s) union_global(parent, p, p + 1);
    if (last_row && i + 1 < M && signs[p + M] == s) union_global(parent, p, p + M);
    if (s == eight) {
        if ((last_col || last_row) && i + 1 < M && j + 1 < M && signs[p + M + 1] == s)
            union_global(parent, p, p + M + 1);
        if ((first_col || last_row) && i + 1 < M && j > 0 && signs[p + M - 1] == s)
            union_global(parent, p, p + M - 1);
    }
}

__global__ void merge_seams(int *parent, const int *pairs, int S, int B, long long per_form) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (long long)S * B) return;
    const long long b = k / S;
    const int s = (int)(k % S);
    union_global(parent, (int)(b * per_form + pairs[2 * s]), (int)(b * per_form + pairs[2 * s + 1]));
}

__global__ void compress(int *parent, long long n) {
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) parent[x] = find_global(parent, (int)x);
}

// ---------------------------------------------------------------------------------------------------------------
// Region trees from the component labels, one block per form; everything after the node passes runs in shared memory.
//   pass 1   roots (label == node) -> component list, sorted by root node; hash map root node -> component
//   pass 2   component sizes (warp-aggregated: lanes with the same label add their weights once; weights are 0/1)
//   regions  component and its antipodal image (antipode of the root node); the root region N is self-antipodal
//   pass 3   region adjacency bitmask from horizontally / vertically neighbouring nodes of different signs
//   tree     one warp: prune small non-root leaves (all current leaves at once, until stable), depths and parents
// Same definitions and region order as the PyTorch path (gpu_trees._trees_from_labels). Forms with more than CMAX
// components are flagged (nreg = -1) for a fallback.
// ---------------------------------------------------------------------------------------------------------------

constexpr int CMAX = 256;  // components (on S^2) per form
constexpr int HS = 512;    // hash slots (power of two, >= 2 CMAX)
constexpr int AW = CMAX / 32;  // words per adjacency row
constexpr int TREE_THREADS = 512;

__device__ __forceinline__ unsigned hslot(int key) { return ((unsigned)key * 2654435761u) >> 23; }  // 9 bits

__device__ __forceinline__ int lookup(const int *hkey, const int *hval, int key) {
    unsigned s = hslot(key);
    while (hkey[s] != key) s = (s + 1) & (HS - 1);
    return hval[s];
}

__device__ __forceinline__ bool bit(const unsigned *w, int r) { return (w[r >> 5] >> (r & 31)) & 1u; }

// Regions from components: {component, antipodal image}, numbered by the smaller component index.
// Inputs per component t < C: cpart (antipodal component), csize, csign. Outputs creg (region of each component),
// rsize, rsign, rroot (self-antipodal) per region; *nr (zeroed by the caller) = number of regions.
__device__ void build_regions(int C, const int *cpart, const int *csize, const int *csign, int *creg, int *rsize,
                              int *rsign, int *rroot, int *nr) {
    for (int t = threadIdx.x; t < C; t += blockDim.x) {
        const int key = min(t, cpart[t]);
        int r = 0;
        for (int u = 0; u < key; ++u) r += u <= cpart[u];
        creg[t] = r;
        if (t == key) {
            const bool self = cpart[t] == t;
            rsize[r] = csize[t] + (self ? 0 : csize[cpart[t]]);
            rsign[r] = csign[t];
            rroot[r] = self;
            atomicAdd(nr, 1);
        }
    }
}

__device__ __forceinline__ void add_edge(unsigned *adj, int ra, int rb) {
    if (ra == rb) return;
    atomicOr(&adj[ra * AW + (rb >> 5)], 1u << (rb & 31));
    atomicOr(&adj[rb * AW + (ra >> 5)], 1u << (ra & 31));
}

// Pruning, depths and parents (one warp), then the per-form outputs; called by all threads of the block.
__device__ void tree_phase(int b, int R, int min_size, const int *rsize, const int *rsign, const int *rroot, int *rpar,
                           volatile int *rdep, const unsigned *adj, unsigned *alive, unsigned *kill, int *nreg_out,
                           int *nov_out, int *ok_out, int *size_out, int *sign_out, int *flags_out, int *parent_out,
                           int *depth_out) {
    const int tid = threadIdx.x, lane = tid & 31;
    if (tid < 32) {
        if (lane < AW) {
            const int lo = lane * 32;
            alive[lane] = R >= lo + 32 ? 0xffffffffu : (R > lo ? (1u << (R - lo)) - 1 : 0u);
            kill[lane] = 0;
        }
        __syncwarp();
        while (true) {
            bool any = false;
            for (int r = lane; r < R; r += 32) {
                if (!bit(alive, r) || rroot[r] || rsize[r] >= min_size) continue;
                int deg = 0;
                for (int w = 0; w < AW; ++w) deg += __popc(adj[r * AW + w] & alive[w]);
                if (deg <= 1) {
                    atomicOr(&kill[r >> 5], 1u << (r & 31));
                    any = true;
                }
            }
            __syncwarp();
            if (!__any_sync(0xffffffffu, any)) break;
            if (lane < AW) {
                alive[lane] &= ~kill[lane];
                kill[lane] = 0;
            }
            __syncwarp();
        }
        for (int r = lane; r < R; r += 32) {
            rdep[r] = rroot[r] && bit(alive, r) ? 0 : -1;
            rpar[r] = -1;
        }
        __syncwarp();
        for (int d = 0;; ++d) {
            bool any = false;
            for (int r = lane; r < R; r += 32) {
                if (!bit(alive, r) || rdep[r] >= 0) continue;
                for (int w = 0; w < AW && rpar[r] < 0; ++w) {
                    unsigned bits = adj[r * AW + w] & alive[w];
                    while (bits) {
                        const int q = w * 32 + __ffs(bits) - 1;
                        bits &= bits - 1;
                        if (rdep[q] == d) {
                            rpar[r] = q;
                            break;
                        }
                    }
                }
                if (rpar[r] >= 0) any = true;
            }
            __syncwarp();
            for (int r = lane; r < R; r += 32)
                if (rpar[r] >= 0 && rdep[r] < 0) rdep[r] = d + 1;
            __syncwarp();
            if (!__any_sync(0xffffffffu, any)) break;
        }
        int n_alive = 0, n_roots = 0, reached = 0, deg2 = 0;
        for (int r = lane; r < R; r += 32) {
            n_roots += rroot[r];
            if (!bit(alive, r)) continue;
            ++n_alive;
            reached += rdep[r] >= 0;
            for (int w = 0; w < AW; ++w) deg2 += __popc(adj[r * AW + w] & alive[w]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            n_alive += __shfl_xor_sync(0xffffffffu, n_alive, o);
            n_roots += __shfl_xor_sync(0xffffffffu, n_roots, o);
            reached += __shfl_xor_sync(0xffffffffu, reached, o);
            deg2 += __shfl_xor_sync(0xffffffffu, deg2, o);
        }
        if (lane == 0) {
            nreg_out[b] = R;
            nov_out[b] = n_alive - 1;
            ok_out[b] = n_roots == 1 && deg2 / 2 == n_alive - 1 && reached == n_alive;
        }
    }
    __syncthreads();
    for (int r = tid; r < R; r += blockDim.x) {
        const long long o = (long long)b * CMAX + r;
        const bool a = bit(alive, r);
        size_out[o] = rsize[r];
        sign_out[o] = rsign[r];
        flags_out[o] = (int)a | ((int)(a && rroot[r]) << 1);
        parent_out[o] = rpar[r];
        depth_out[o] = rdep[r];
    }
}


__global__ void __launch_bounds__(TREE_THREADS) form_trees(const uint8_t *signs, const int *labels, const int *weight,
                                                           const int *anti, int M, int min_size, int *nreg_out,
                                                           int *nov_out, int *ok_out, int *size_out, int *sign_out,
                                                           int *flags_out, int *parent_out, int *depth_out) {
    __shared__ int hkey[HS], hval[HS];
    __shared__ int tmp[CMAX], croot[CMAX], csize[CMAX], cpart[CMAX], creg[CMAX];
    __shared__ int rsize[CMAX], rsign[CMAX], rroot[CMAX], rpar[CMAX];
    __shared__ volatile int rdep[CMAX];
    __shared__ unsigned adj[CMAX * AW];
    __shared__ unsigned alive[AW], kill[AW];
    __shared__ int nc, nr;

    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int V = 6 * M * M;
    const long long base = (long long)b * V;
    const int *lab = labels + base;
    const uint8_t *sg = signs + base;
    const int ib = (int)base;  // labels are global node indices (int32 by the caller's check)

    for (int k = tid; k < HS; k += blockDim.x) hkey[k] = -1;
    for (int k = tid; k < CMAX * AW; k += blockDim.x) adj[k] = 0;
    for (int k = tid; k < CMAX; k += blockDim.x) csize[k] = 0;
    if (tid == 0) nc = nr = 0;
    __syncthreads();

    // pass 1: roots
    for (int p = tid; p < V; p += blockDim.x)
        if (lab[p] == ib + p) {
            const int k = atomicAdd(&nc, 1);
            if (k < CMAX) tmp[k] = p;
        }
    __syncthreads();
    const int C = nc;
    if (C > CMAX) {
        if (tid == 0) nreg_out[b] = -1;
        return;
    }
    for (int t = tid; t < C; t += blockDim.x) {
        int r = 0;
        for (int u = 0; u < C; ++u) r += tmp[u] < tmp[t];
        croot[r] = tmp[t];
    }
    __syncthreads();
    for (int t = tid; t < C; t += blockDim.x) {
        unsigned s = hslot(croot[t]);
        while (atomicCAS(&hkey[s], -1, croot[t]) != -1) s = (s + 1) & (HS - 1);
        hval[s] = t;
    }
    __syncthreads();

    // pass 2: sizes (all lanes of a warp take part in every iteration, as required by the warp intrinsics)
    for (int p0 = 0; p0 < V; p0 += blockDim.x) {
        const int p = p0 + tid;
        const bool in = p < V;
        const int l = in ? lab[p] - ib : -1;
        const unsigned grp = __match_any_sync(0xffffffffu, l);
        const unsigned wb = __ballot_sync(0xffffffffu, in && weight[p] != 0);
        if (in && lane == __ffs(grp) - 1) {
            const int cnt = __popc(grp & wb);
            if (cnt) atomicAdd(&csize[lookup(hkey, hval, l)], cnt);
        }
    }
    __syncthreads();

    // regions: {component, antipodal image}, numbered by the smaller component index
    for (int t = tid; t < C; t += blockDim.x) {
        cpart[t] = lookup(hkey, hval, lab[anti[croot[t]]] - ib);
        tmp[t] = 2 * (int)sg[croot[t]] - 1;  // component sign (tmp is free after sorting)
    }
    __syncthreads();
    build_regions(C, cpart, csize, tmp, creg, rsize, rsign, rroot, &nr);
    __syncthreads();
    const int R = nr;

    // pass 3: region adjacency
    for (int p = tid; p < V; p += blockDim.x) {
        const int j = p % M, i = (p / M) % M;
        const uint8_t s = sg[p];
        const bool h = j + 1 < M && sg[p + 1] != s, v = i + 1 < M && sg[p + M] != s;
        if (!h && !v) continue;
        const int ra = creg[lookup(hkey, hval, lab[p] - ib)];
        for (int e = 0; e < 2; ++e) {
            if (!(e ? v : h)) continue;
            const int rb = creg[lookup(hkey, hval, lab[p + (e ? M : 1)] - ib)];
            add_edge(adj, ra, rb);
        }
    }
    __syncthreads();

    tree_phase(b, R, min_size, rsize, rsign, rroot, rpar, rdep, adj, alive, kill, nreg_out, nov_out, ok_out, size_out,
               sign_out, flags_out, parent_out, depth_out);
}

// ---------------------------------------------------------------------------------------------------------------
// Fused per-form kernel: coefficients -> signs -> union-find over row runs -> region trees, all in shared memory.
//
// Rows. On face f = 2 axis + side, row i, the points are x_axis = +-N, x_o0 = i - N, x_o1 = t = j - N (o0 < o1 the
// other axes; as in euler.face_grid_points). The form restricted to a row is a univariate polynomial of degree <= deg
// in t, so a row has at most deg sign changes (Bezout; a grid row lies on a projective line), i.e. <= deg + 1 runs.
// Its coefficients are computed per row in float32 together with sums of absolute terms, and the M points evaluated
// by float32 Horner; values within the rounding bound are recomputed in float64 (sign = value >= 0 at the integer
// points; a positive rescaling of an even-degree form does not change signs).
// Runs. One warp per row: ballots give the sign changes; a run is stored as its first column j0 (signs alternate
// along a row, rsgn = sign at j = 0). Union-find over runs: same-sign runs of adjacent rows whose column intervals
// overlap (4-connectivity), or touch diagonally for the class `eight`; duplicated cube-edge points (seams).
// The smallest run index of a component holds its smallest node, so components and regions come out in the same
// order as in the node-based path (form_trees). Region sizes count canonical nodes (weights); region adjacency from
// horizontally / vertically neighbouring runs of different signs. Overflow: nreg = -1 (> CMAX components), -2 (more
// runs than the shared-memory capacity rcap).
// ---------------------------------------------------------------------------------------------------------------

constexpr int FUSED_THREADS = 512;
constexpr int MAXDEG = 12;

// exclusive scan in place over a[0..n) (int or short); returns the total. All threads must call it.
template <typename V>
__device__ int block_scan(V *a, int n, int *wsum) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nt = blockDim.x;
    const int per = (n + nt - 1) / nt, lo = min(n, tid * per), hi = min(n, lo + per);
    int local = 0;
    for (int k = lo; k < hi; ++k) local += a[k];
    int x = local;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < (nt >> 5) ? wsum[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        wsum[lane] = w;  // inclusive warp totals
    }
    __syncthreads();
    int run = x - local + (warp ? wsum[warp - 1] : 0);
    for (int k = lo; k < hi; ++k) {
        const int v = a[k];
        a[k] = (V)run;
        run += v;
    }
    const int total = wsum[(nt >> 5) - 1];
    __syncthreads();
    return total;
}

__device__ __forceinline__ double ipow(double x, int e) {
    double r = 1.0;
    for (int k = 0; k < e; ++k) r *= x;
    return r;
}

// Row polynomials. Per face f, the bivariate coefficients G_f[e][k] = c_m (+-N)^(e_axis) of x_o0^e x_o1^k (and
// their absolute values A_f) are precomputed in shared memory (float64 products rounded to float32). On row i, lane
// k <= deg evaluates q[k] = sum_e G_f[e][k] u^e and Q[k] = sum_e A_f[e][k] |u|^e (u = i - N) by Horner; the results
// are broadcast to all lanes. Arrays in registers are indexed with compile-time indices only (unrolled to MAXDEG).
constexpr int GS = MAXDEG + 1;

struct RowPoly {
    float q[MAXDEG + 1], Q[MAXDEG + 1];
    double xa, xo;
    int axis, o0, o1;
};

__device__ __forceinline__ void row_poly(const float *G, const float *A, int deg, int N, int f, int i, RowPoly &rp) {
    rp.axis = f >> 1;
    rp.o0 = rp.axis == 0 ? 1 : 0;
    rp.o1 = rp.axis == 2 ? 1 : 2;
    rp.xa = (f & 1) ? N : -N;
    rp.xo = i - N;
    const int lane = threadIdx.x & 31;
    const float u = (float)(i - N), au = fabsf(u);
    float qk = 0.f, Qk = 0.f;
    if (lane <= deg) {
        const float *g = G + f * GS * GS, *ab = A + f * GS * GS;
        for (int e = deg - lane; e >= 0; --e) {  // e + k <= deg
            qk = fmaf(qk, u, g[e * GS + lane]);
            Qk = fmaf(Qk, au, ab[e * GS + lane]);
        }
    }
#pragma unroll
    for (int k = 0; k <= MAXDEG; ++k) {
        rp.q[k] = __shfl_sync(0xffffffffu, qk, k);
        rp.Q[k] = __shfl_sync(0xffffffffu, Qk, k);
    }
}

// sign (value >= 0) of the form at (row, t = j - N): float32 Horner with a rounding bound; values within the bound
// are recomputed in float64 from the coefficients (rare)
__device__ __forceinline__ bool row_sign(const RowPoly &rp, const double *c64, const int *ex, int D, int deg, int N,
                                         int j, float tolf) {
    const float t = (float)(j - N), at = fabsf(t);
    float v = 0.f, S = 0.f;
#pragma unroll
    for (int k = MAXDEG; k >= 0; --k) {
        if (k <= deg) {
            v = fmaf(v, t, rp.q[k]);
            S = fmaf(S, at, rp.Q[k]);
        }
    }
    if (fabsf(v) > tolf * S) return v >= 0.f;
    const double td = j - N;
    double w = 0.0;
    for (int m = 0; m < D; ++m)
        w += c64[m] * ipow(rp.xa, ex[3 * m + rp.axis]) * ipow(rp.xo, ex[3 * m + rp.o0]) * ipow(td, ex[3 * m + rp.o1]);
    return w >= 0.0;
}

// last run of row r (runs rs..re-1) whose first column is <= j
__device__ __forceinline__ int find_run(const short *j0, int rs, int re, int j) {
    int k = rs;
    while (k + 1 < re && j0[k + 1] <= j) ++k;
    return k;
}

__device__ __forceinline__ void union_runs(int *par, int a, int b) {
    while (true) {
        a = find_splitting(par, a);
        b = find_splitting(par, b);
        if (a == b) return;
        if (a < b) {
            const int t = a;
            a = b;
            b = t;
        }
        const int old = atomicMin(&par[a], b);
        if (old == a) return;
        a = old;
    }
}

__global__ void __launch_bounds__(FUSED_THREADS) fused_trees(const double *coefs, int D, const int *expo, int deg,
                                                             int N, int eight, int min_size, int rcap, float tolf,
                                                             const int *seams, int n_seams, const int *weight,
                                                             int *nreg_out, int *nov_out, int *ok_out, int *size_out,
                                                             int *sign_out, int *flags_out, int *parent_out,
                                                             int *depth_out, long long *prof) {
    extern __shared__ __align__(16) unsigned char smem[];
#define PROF(k) \
    if (prof && threadIdx.x == 0) prof[(long long)blockIdx.x * 8 + (k)] = clock64()
    __shared__ double c[64];
    __shared__ float G[6 * GS * GS], A[6 * GS * GS];
    __shared__ int ex[192];
    __shared__ int wsum[32];
    __shared__ int croot[CMAX], csize[CMAX], cpart[CMAX], csgn[CMAX], creg[CMAX];
    __shared__ int rsize[CMAX], rsign[CMAX], rroot[CMAX], rpar[CMAX];
    __shared__ volatile int rdep[CMAX];
    __shared__ unsigned adj[CMAX * AW];
    __shared__ unsigned alive[AW], kill[AW];
    __shared__ int nr, total_runs, n_comp;

    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    const int M = 2 * N + 1, NR = 6 * M;
    int *rstart = reinterpret_cast<int *>(smem);  // NR + 1
    int *par = rstart + NR + 1;                    // rcap
    short *j0 = reinterpret_cast<short *>(par + rcap);  // rcap
    short *cofr = j0 + rcap;                       // rcap: component of each run
    unsigned char *rsgn = reinterpret_cast<unsigned char *>(cofr + rcap);  // NR

    for (int k = tid; k < D; k += blockDim.x) c[k] = coefs[(long long)b * D + k];
    for (int k = tid; k < 3 * D; k += blockDim.x) ex[k] = expo[k];
    for (int k = tid; k < 6 * GS * GS; k += blockDim.x) G[k] = A[k] = 0.f;
    __syncthreads();
    for (int k = tid; k < 6 * D; k += blockDim.x) {  // face coefficients (each (e, k) pair has one monomial per face)
        const int f = k / D, m = k % D, axis = f >> 1, o0 = axis == 0 ? 1 : 0, o1 = axis == 2 ? 1 : 2;
        const double g = c[m] * ipow((f & 1) ? N : -N, ex[3 * m + axis]);
        G[f * GS * GS + ex[3 * m + o0] * GS + ex[3 * m + o1]] = (float)g;
        A[f * GS * GS + ex[3 * m + o0] * GS + ex[3 * m + o1]] = (float)fabs(g);
    }
    for (int k = tid; k < CMAX * AW; k += blockDim.x) adj[k] = 0;
    for (int k = tid; k < CMAX; k += blockDim.x) csize[k] = 0;
    if (tid == 0) nr = 0;
    __syncthreads();

    PROF(0);
    // phase 1: runs per row (one warp per row)
    for (int r = warp; r < NR; r += nwarps) {
        RowPoly rp;
        row_poly(G, A, deg, N, r / M, r % M, rp);
        int changes = 0;
        unsigned carry = 0;
        for (int base = 0; base < M; base += 32) {
            const int j = base + lane;
            const bool sj = j < M && row_sign(rp, c, ex, D, deg, N, j, tolf);
            const unsigned m = __ballot_sync(0xffffffffu, sj);
            const unsigned valid = (M - base >= 32) ? 0xffffffffu : ((1u << (M - base)) - 1);
            unsigned ch = (m ^ ((m << 1) | carry)) & valid;
            if (base == 0) {
                ch &= ~1u;
                if (lane == 0) rsgn[r] = sj;
            }
            changes += __popc(ch);
            carry = (m >> 31) & 1u;
        }
        if (lane == 0) rstart[r] = changes + 1;
    }
    __syncthreads();
    const int R = block_scan(rstart, NR, wsum);
    if (tid == 0) rstart[NR] = R;
    if (R > rcap) {
        if (tid == 0) nreg_out[b] = -2;
        return;
    }
    PROF(1);
    // phase 2: run starts
    for (int r = warp; r < NR; r += nwarps) {
        RowPoly rp;
        row_poly(G, A, deg, N, r / M, r % M, rp);
        int k0 = rstart[r] + 1;
        if (lane == 0) j0[rstart[r]] = 0;
        unsigned carry = 0;
        for (int base = 0; base < M; base += 32) {
            const int j = base + lane;
            const bool sj = j < M && row_sign(rp, c, ex, D, deg, N, j, tolf);
            const unsigned m = __ballot_sync(0xffffffffu, sj);
            const unsigned valid = (M - base >= 32) ? 0xffffffffu : ((1u << (M - base)) - 1);
            unsigned ch = (m ^ ((m << 1) | carry)) & valid;
            if (base == 0) ch &= ~1u;
            if ((ch >> lane) & 1u) j0[k0 + __popc(ch & ((1u << lane) - 1))] = (short)j;
            k0 += __popc(ch);
            carry = (m >> 31) & 1u;
        }
    }
    for (int k = tid; k < R; k += blockDim.x) par[k] = k;
    __syncthreads();

    PROF(2);
    // phase 3: unions
    for (int rr = tid; rr < 6 * (M - 1); rr += blockDim.x) {  // adjacent rows within a face
        const int f = rr / (M - 1), i = rr % (M - 1), r = f * M + i;
        const int as = rstart[r], ae = rstart[r + 1], bs = ae, be = rstart[r + 2];
        int kb0 = bs;
        for (int ka = as; ka < ae; ++ka) {
            const int a0 = j0[ka], a1 = ka + 1 < ae ? j0[ka + 1] - 1 : M - 1;
            const int sa = rsgn[r] ^ ((ka - as) & 1);
            const int ext = sa == eight ? 1 : 0;
            while (kb0 < be && (kb0 + 1 < be ? j0[kb0 + 1] - 1 : M - 1) < a0 - ext) ++kb0;
            for (int kb = kb0; kb < be && j0[kb] <= a1 + ext; ++kb) {
                if ((rsgn[r + 1] ^ ((kb - bs) & 1)) == sa) union_runs(par, ka, kb);
            }
        }
    }
    for (int s = tid; s < n_seams; s += blockDim.x) {  // duplicated cube-edge points
        const int p = seams[2 * s], q = seams[2 * s + 1];
        const int rp = p / M, rq = q / M;
        union_runs(par, find_run(j0, rstart[rp], rstart[rp + 1], p % M), find_run(j0, rstart[rq], rstart[rq + 1], q % M));
    }
    __syncthreads();
    for (int k = tid; k < R; k += blockDim.x) cofr[k] = (short)find_shared(par, k);  // roots, then compress
    __syncthreads();
    for (int k = tid; k < R; k += blockDim.x) par[k] = cofr[k];
    __syncthreads();

    PROF(3);
    // phase 4: components (ordered by root run), sizes, antipodes, signs
    for (int k = tid; k < R; k += blockDim.x) cofr[k] = par[k] == k;
    __syncthreads();
    const int C = block_scan(cofr, R, wsum);
    if (C > CMAX) {
        if (tid == 0) nreg_out[b] = -1;
        return;
    }
    for (int k = tid; k < R; k += blockDim.x)
        if (par[k] == k) croot[cofr[k]] = k;
    __syncthreads();
    for (int k = tid; k < R; k += blockDim.x)
        if (par[k] != k) cofr[k] = cofr[par[k]];
    __syncthreads();
    // run -> row: binary search in rstart
    auto row_of = [&](int k) {
        int lo = 0, hi = NR - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (rstart[mid] <= k) lo = mid; else hi = mid - 1;
        }
        return lo;
    };
    for (int k = tid; k < R; k += blockDim.x) {
        const int r = row_of(k), f = r / M, i = r % M, base = (f * M + i) * M;
        const int a0 = j0[k], a1 = k + 1 < rstart[r + 1] ? j0[k + 1] - 1 : M - 1;
        int w = 0;
        if (i == 0 || i == M - 1) {
            for (int j = a0; j <= a1; ++j) w += weight[base + j];
        } else {
            w = a1 - a0 + 1 - (a0 == 0 && weight[base] == 0) - (a1 == M - 1 && weight[base + M - 1] == 0);
        }
        if (w) atomicAdd(&csize[cofr[k]], w);
    }
    for (int t = tid; t < C; t += blockDim.x) {
        const int k = croot[t], r = row_of(k), f = r / M, i = r % M;
        const int ra = (f ^ 1) * M + (M - 1 - i);
        cpart[t] = cofr[find_run(j0, rstart[ra], rstart[ra + 1], M - 1 - j0[k])];
        csgn[t] = 2 * (rsgn[r] ^ ((k - rstart[r]) & 1)) - 1;
    }
    __syncthreads();
    build_regions(C, cpart, csize, csgn, creg, rsize, rsign, rroot, &nr);
    __syncthreads();
    const int NRG = nr;

    PROF(4);
    // phase 5: region adjacency (horizontal: consecutive runs; vertical: overlapping runs of adjacent rows)
    for (int k = tid; k < R; k += blockDim.x) {
        const int r = row_of(k);
        if (k + 1 < rstart[r + 1]) add_edge(adj, creg[cofr[k]], creg[cofr[k + 1]]);
    }
    for (int rr = tid; rr < 6 * (M - 1); rr += blockDim.x) {
        const int f = rr / (M - 1), i = rr % (M - 1), r = f * M + i;
        const int as = rstart[r], ae = rstart[r + 1], bs = ae, be = rstart[r + 2];
        int kb0 = bs;
        for (int ka = as; ka < ae; ++ka) {
            const int a0 = j0[ka], a1 = ka + 1 < ae ? j0[ka + 1] - 1 : M - 1;
            const int sa = rsgn[r] ^ ((ka - as) & 1);
            while (kb0 < be && (kb0 + 1 < be ? j0[kb0 + 1] - 1 : M - 1) < a0) ++kb0;
            for (int kb = kb0; kb < be && j0[kb] <= a1; ++kb)
                if ((rsgn[r + 1] ^ ((kb - bs) & 1)) != sa) add_edge(adj, creg[cofr[ka]], creg[cofr[kb]]);
        }
    }
    __syncthreads();
    PROF(5);
    tree_phase(b, NRG, min_size, rsize, rsign, rroot, rpar, rdep, adj, alive, kill, nreg_out, nov_out, ok_out,
               size_out, sign_out, flags_out, parent_out, depth_out);
    PROF(6);
#undef PROF
}

}  // namespace

int ccl_tree_capacity() { return CMAX; }

// Host launcher for form_trees: outputs nreg/nov/ok (B,), per-region arrays (B, CMAX); returns a CUDA error code.
int ccl_form_trees(const uint8_t *signs, const int *labels, const int *weight, const int *anti, int B, int M,
                   int min_size, int *nreg, int *nov, int *ok, int *size, int *sign, int *flags, int *parent, int *depth,
                   cudaStream_t stream) {
    form_trees<<<B, TREE_THREADS, 0, stream>>>(signs, labels, weight, anti, M, min_size, nreg, nov, ok, size, sign,
                                               flags, parent, depth);
    return (int)cudaGetLastError();
}

// Shared memory (bytes) of fused_trees for a given N and run capacity.
static size_t fused_smem(int N, int rcap) {
    const int NR = 6 * (2 * N + 1);
    return (size_t)(NR + 1) * 4 + (size_t)rcap * 8 + NR + 16;
}

// Run capacity of fused_trees: deg + 1 runs per row (Bezout) plus slack, limited by the shared memory per block.
int ccl_fused_capacity(int N, int deg) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, fused_trees);
    const int NR = 6 * (2 * N + 1);
    const long long budget = (long long)optin - (long long)attr.sharedSizeBytes - (NR + 1) * 4 - NR - 64;
    long long rcap = std::min<long long>((long long)NR * (deg + 2), budget / 8);
    return (int)std::min<long long>(rcap, 32767);
}

// Host launcher for fused_trees: coefs (B, D) float64, expo (D, 3) int; outputs as ccl_form_trees.
int ccl_fused_trees(const double *coefs, int D, const int *expo, int deg, int N, int eight, int min_size, int rcap,
                    const int *seams, int n_seams, const int *weight, int B, int *nreg, int *nov, int *ok, int *size,
                    int *sign, int *flags, int *parent, int *depth, long long *prof, cudaStream_t stream) {
    const size_t smem = fused_smem(N, rcap);
    // float32 rounding bound: |v32 - v| <= (powers deg + D terms + Horner 2 deg + 8) u S, with a safety factor 1.5
    const float tolf = 1.5f * (3 * deg + D + 8) * 5.9604645e-8f;
    cudaFuncSetAttribute(fused_trees, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    fused_trees<<<B, FUSED_THREADS, smem, stream>>>(coefs, D, expo, deg, N, eight, min_size, rcap, tolf, seams, n_seams,
                                                    weight, nreg, nov, ok, size, sign, flags, parent, depth, prof);
    return (int)cudaGetLastError();
}

// Host launcher: returns a CUDA error code (0 on success).
int ccl_label_faces(const uint8_t *signs, int *parent, const int *seam_pairs, int n_seams, int B, int M, int eight,
                    cudaStream_t stream) {
    const dim3 block(T, T);
    const dim3 grid((M + T - 1) / T, (M + T - 1) / T, 6 * B);
    local_tiles<<<grid, block, 0, stream>>>(signs, parent, M, eight);
    // tile-border nodes of one face (cached per M)
    static int cached_M = -1, nb = 0;
    static int *border = nullptr;
    if (cached_M != M) {
        std::vector<int> list;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                if (j % T == T - 1 || j % T == 0 || i % T == T - 1) list.push_back(i * M + j);
        if (border) cudaFree(border);
        nb = (int)list.size();
        cudaMalloc(&border, nb * sizeof(int));
        cudaMemcpy(border, list.data(), nb * sizeof(int), cudaMemcpyHostToDevice);
        cached_M = M;
    }
    const long long nk = 6LL * B * nb;
    border_merge<<<(unsigned)((nk + 255) / 256), 256, 0, stream>>>(signs, parent, border, nb, 6LL * B, M, eight);
    const long long per_form = 6LL * M * M;
    if (n_seams > 0) {
        const long long k = (long long)n_seams * B;
        merge_seams<<<(unsigned)((k + 255) / 256), 256, 0, stream>>>(parent, seam_pairs, n_seams, B, per_form);
    }
    const long long n = per_form * B;
    compress<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(parent, n);
    return (int)cudaGetLastError();
}
