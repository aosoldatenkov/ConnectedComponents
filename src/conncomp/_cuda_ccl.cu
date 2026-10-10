// Block-based union-find connected-component labelling on the six cube-face grids (CUDA).
//
// Layout: for each form b and face f (fb = 6 b + f) an M x M grid; node index ((fb * M) + i) * M + j.
// Connectivity: 4-neighbours for both sign classes, diagonals only for the class `eight` (8/4 digital topology).
// Points on cube edges/corners are duplicated on 2/3 faces; `seams` lists (node, canonical node) pairs per form.
//
// Kernels:
//   local_tiles   T x T tiles labelled in shared memory (union-find with atomicMin), roots written to global memory
//   border_merge  global union-find across tile borders (each grid edge is handled by exactly one of the kernels)
//   merge_seams   union of the duplicated cube-edge points
//   compress      parent[x] = root(x) with read-only root finding (no halving writes: they would race)
//   form_trees    region trees from the labels, one block per form (see below)

#include <cstdint>
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

__device__ __forceinline__ void union_shared(int *lab, int a, int b) {
    while (true) {
        a = find_shared(lab, a);
        b = find_shared(lab, b);
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

__global__ void border_merge(const uint8_t *signs, int *parent, int M, int eight) {
    const int fb = blockIdx.z;
    const int i = blockIdx.y * T + threadIdx.y, j = blockIdx.x * T + threadIdx.x;
    if (i >= M || j >= M) return;
    const bool last_col = (j % T) == T - 1, last_row = (i % T) == T - 1, first_col = (j % T) == 0;
    if (!last_col && !last_row && !first_col) return;
    const long long base = (long long)fb * M * M;
    const int p = (int)(base + (long long)i * M + j);
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
    for (int t = tid; t < C; t += blockDim.x) cpart[t] = lookup(hkey, hval, lab[anti[croot[t]]] - ib);
    __syncthreads();
    for (int t = tid; t < C; t += blockDim.x) {
        const int key = min(t, cpart[t]);
        int r = 0;
        for (int u = 0; u < key; ++u) r += u <= cpart[u];
        creg[t] = r;
        if (t == key) {
            const bool self = cpart[t] == t;
            rsize[r] = csize[t] + (self ? 0 : csize[cpart[t]]);
            rsign[r] = 2 * (int)sg[croot[t]] - 1;
            rroot[r] = self;
            atomicAdd(&nr, 1);
        }
    }
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
            if (ra == rb) continue;
            atomicOr(&adj[ra * AW + (rb >> 5)], 1u << (rb & 31));
            atomicOr(&adj[rb * AW + (ra >> 5)], 1u << (ra & 31));
        }
    }
    __syncthreads();

    // tree: one warp
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

// Host launcher: returns a CUDA error code (0 on success).
int ccl_label_faces(const uint8_t *signs, int *parent, const int *seam_pairs, int n_seams, int B, int M, int eight,
                    cudaStream_t stream) {
    const dim3 block(T, T);
    const dim3 grid((M + T - 1) / T, (M + T - 1) / T, 6 * B);
    local_tiles<<<grid, block, 0, stream>>>(signs, parent, M, eight);
    border_merge<<<grid, block, 0, stream>>>(signs, parent, M, eight);
    const long long per_form = 6LL * M * M;
    if (n_seams > 0) {
        const long long k = (long long)n_seams * B;
        merge_seams<<<(unsigned)((k + 255) / 256), 256, 0, stream>>>(parent, seam_pairs, n_seams, B, per_form);
    }
    const long long n = per_form * B;
    compress<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(parent, n);
    return (int)cudaGetLastError();
}
