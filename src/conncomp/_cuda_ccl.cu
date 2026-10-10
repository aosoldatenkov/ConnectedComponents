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

}  // namespace

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
