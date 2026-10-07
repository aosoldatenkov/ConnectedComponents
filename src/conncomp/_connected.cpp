// Connected components of the sign regions {F >= 0} and {F < 0} on a W x W grid with a general
// neighbour pattern (pat[k, :, i, j] = k-th neighbour of pixel (i, j), 12 neighbours), as used by
// conncomp.grid for the RP^2 model (8 adjacent pixels + 4 pixels around the antipodal point).
//
// Two pixels are connected if one is in the other's neighbour list and they have the same sign
// class; the neighbour relation is treated as undirected. Input is either float64 values (class:
// v >= 0) or uint8 sign classes (0 / 1), the latter being 8x cheaper to transfer from the GPU.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

// ------------------------------------------------------------
// Union-Find (Disjoint Set Union) over pixels
// ------------------------------------------------------------
struct DSU {
    std::vector<int> parent, size;

    explicit DSU(int n) : parent(n), size(n, 1) { reset(); }

    void reset() {
        for (int i = 0; i < (int)parent.size(); ++i) parent[i] = i;
        std::fill(size.begin(), size.end(), 1);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
};

template <typename T> inline int sign_class(T v) { return v >= 0; }
template <> inline int sign_class<uint8_t>(uint8_t v) { return v != 0; }

// Union all same-class neighbour pairs of a single W x W array.
// pat layout: pat[k * 2 * W * W + d * W * W + i * W + j], k < 12, d in {0, 1}
template <typename T>
static void unite_single(int W, const T *vals, const int *pat, DSU &dsu) {
    const int P = W * W;
    for (int p = 0; p < P; ++p) {
        const int c = sign_class(vals[p]);
        for (int k = 0; k < 12; ++k) {
            const int q = pat[k * 2 * P + p] * W + pat[k * 2 * P + P + p];
            // the 8 adjacent neighbours (k < 8) are symmetric: visit each such edge from one side only;
            // the antipodal lists (k >= 8) are not symmetric, so those edges are used from both sides
            if (q == p || (k < 8 && q < p)) continue;
            if (sign_class(vals[q]) == c) dsu.unite(p, q);
        }
    }
}

// Component sizes, in the order of increasing root index
static std::vector<int> sizes_of(DSU &dsu, int P) {
    std::vector<int> out;
    for (int p = 0; p < P; ++p)
        if (dsu.find(p) == p) out.push_back(dsu.size[p]);
    return out;
}

static int count_of(DSU &dsu, int P, int min_size) {
    int n = 0;
    for (int p = 0; p < P; ++p)
        if (dsu.parent[p] == p && dsu.size[p] >= min_size) ++n;
    return n;
}

static void check_pat(const py::buffer_info &pb, int W) {
    if (pb.ndim != 4 || (int)pb.shape[0] != 12 || (int)pb.shape[1] != 2 || (int)pb.shape[2] != W ||
        (int)pb.shape[3] != W)
        throw std::runtime_error("pat must have shape (12, 2, W, W) with the same W as the values");
}

// Run fn(index, dsu) for index in [0, N) on num_threads threads (GIL released), one DSU per thread.
template <typename Fn>
static void parallel_for(int N, int P, int num_threads, Fn fn) {
    unsigned hw = std::thread::hardware_concurrency();
    int tcount = num_threads <= 0 ? (hw ? (int)hw : 1) : num_threads;
    tcount = std::max(1, std::min(tcount, N));
    py::gil_scoped_release release;
    std::vector<std::thread> threads;
    for (int t = 0; t < tcount; ++t) {
        int s = (int)((long long)N * t / tcount), e = (int)((long long)N * (t + 1) / tcount);
        threads.emplace_back([s, e, P, &fn]() {
            DSU dsu(P);
            for (int idx = s; idx < e; ++idx) {
                dsu.reset();
                fn(idx, dsu);
            }
        });
    }
    for (auto &th : threads) th.join();
}

// ------------------------------------------------------------
// Single arrays
// ------------------------------------------------------------
py::list components(py::array_t<double, py::array::c_style | py::array::forcecast> vals,
                    py::array_t<int, py::array::c_style | py::array::forcecast> pat) {
    py::buffer_info vb = vals.request(), pb = pat.request();
    if (vb.ndim != 2 || vb.shape[0] != vb.shape[1]) throw std::runtime_error("vals must be a 2D array (W, W)");
    const int W = (int)vb.shape[0];
    check_pat(pb, W);
    DSU dsu(W * W);
    unite_single(W, static_cast<const double *>(vb.ptr), static_cast<const int *>(pb.ptr), dsu);
    py::list out;
    for (int s : sizes_of(dsu, W * W)) out.append(s);
    return out;
}

// Returns (labels[W, W], sizes): labels[i, j] is the index of the component of pixel (i, j) in sizes
// (same order as the output of components / components_batch).
py::tuple component_labels(py::array_t<double, py::array::c_style | py::array::forcecast> vals,
                           py::array_t<int, py::array::c_style | py::array::forcecast> pat) {
    py::buffer_info vb = vals.request(), pb = pat.request();
    if (vb.ndim != 2 || vb.shape[0] != vb.shape[1]) throw std::runtime_error("vals must be a 2D array (W, W)");
    const int W = (int)vb.shape[0], P = W * W;
    check_pat(pb, W);
    DSU dsu(P);
    unite_single(W, static_cast<const double *>(vb.ptr), static_cast<const int *>(pb.ptr), dsu);
    std::vector<int> index(P, -1);
    py::list sizes;
    int next = 0;
    for (int p = 0; p < P; ++p)
        if (dsu.find(p) == p) {
            index[p] = next++;
            sizes.append(dsu.size[p]);
        }
    py::array_t<int> labels({W, W});
    auto L = labels.mutable_unchecked<2>();
    for (int p = 0; p < P; ++p) L(p / W, p % W) = index[dsu.find(p)];
    return py::make_tuple(labels, sizes);
}

// ------------------------------------------------------------
// Batches (parallel)
// ------------------------------------------------------------
template <typename T>
static py::list sizes_batch(py::array_t<T, py::array::c_style | py::array::forcecast> vals,
                            py::array_t<int, py::array::c_style | py::array::forcecast> pat, int num_threads) {
    py::buffer_info vb = vals.request(), pb = pat.request();
    if (vb.ndim != 3 || vb.shape[1] != vb.shape[2]) throw std::runtime_error("values must have shape (N, W, W)");
    const int N = (int)vb.shape[0], W = (int)vb.shape[1], P = W * W;
    check_pat(pb, W);
    const T *v = static_cast<const T *>(vb.ptr);
    const int *pp = static_cast<const int *>(pb.ptr);
    std::vector<std::vector<int>> results(N);
    if (N > 0)
        parallel_for(N, P, num_threads, [&](int idx, DSU &dsu) {
            unite_single(W, v + (size_t)idx * P, pp, dsu);
            results[idx] = sizes_of(dsu, P);
        });
    py::list out;
    for (auto &r : results) {
        py::list d;
        for (int s : r) d.append(s);
        out.append(d);
    }
    return out;
}

py::array_t<int> count_components_batch(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> signs,
                                        py::array_t<int, py::array::c_style | py::array::forcecast> pat,
                                        int min_size, int num_threads) {
    py::buffer_info sb = signs.request(), pb = pat.request();
    if (sb.ndim != 3 || sb.shape[1] != sb.shape[2]) throw std::runtime_error("signs must have shape (N, W, W)");
    const int N = (int)sb.shape[0], W = (int)sb.shape[1], P = W * W;
    check_pat(pb, W);
    const uint8_t *s = static_cast<const uint8_t *>(sb.ptr);
    const int *pp = static_cast<const int *>(pb.ptr);
    py::array_t<int> out(N);
    int *o = out.mutable_data();
    if (N > 0)
        parallel_for(N, P, num_threads, [&](int idx, DSU &dsu) {
            unite_single(W, s + (size_t)idx * P, pp, dsu);
            o[idx] = count_of(dsu, P, min_size);
        });
    return out;
}

// ------------------------------------------------------------
// Planar windows: components that do not touch the border
// ------------------------------------------------------------
// For each (H, W) uint8 sign image of a batch: the number of components with at least min_size
// pixels that do not touch the image border (for a curve in an affine window: its compact ovals,
// one bounded complementary region per oval). The class `eight` (0 or 1) uses 8-connectivity and the
// other class 4-connectivity (a consistent digital topology on the square grid).
py::array_t<int> count_bounded_batch(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> signs,
                                     int min_size, int num_threads, int eight) {
    py::buffer_info sb = signs.request();
    if (sb.ndim != 3) throw std::runtime_error("signs must have shape (N, H, W)");
    const int N = (int)sb.shape[0], H = (int)sb.shape[1], W = (int)sb.shape[2], P = H * W;
    const uint8_t *s = static_cast<const uint8_t *>(sb.ptr);
    py::array_t<int> out(N);
    int *o = out.mutable_data();
    if (N > 0)
        parallel_for(N, P, num_threads, [&](int idx, DSU &dsu) {
            const uint8_t *img = s + (size_t)idx * P;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    const int p = i * W + j, c = img[p];
                    if (j + 1 < W && img[p + 1] == c) dsu.unite(p, p + 1);
                    if (i + 1 < H && img[p + W] == c) dsu.unite(p, p + W);
                    if (c == eight && i + 1 < H && j + 1 < W && img[p + W + 1] == c) dsu.unite(p, p + W + 1);
                    if (c == eight && i + 1 < H && j > 0 && img[p + W - 1] == c) dsu.unite(p, p + W - 1);
                }
            std::vector<char> border(P, 0);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    if (i == 0 || j == 0 || i == H - 1 || j == W - 1) border[dsu.find(i * W + j)] = 1;
            int n = 0;
            for (int p = 0; p < P; ++p)
                if (dsu.parent[p] == p && !border[p] && dsu.size[p] >= min_size) ++n;
            o[idx] = n;
        });
    return out;
}

// ------------------------------------------------------------
// Pybind11 module definition
// ------------------------------------------------------------
PYBIND11_MODULE(_connected, m) {
    m.doc() = "Connected components of sign regions on grids with general neighbour patterns (union-find)";
    m.def("components", &components, py::arg("vals"), py::arg("pat"),
          "Component sizes of the sign regions of a single (W, W) float64 array.");
    m.def("component_labels", &component_labels, py::arg("vals"), py::arg("pat"),
          "Component labels of a single array: returns (labels[W, W], sizes).");
    m.def("components_batch", &sizes_batch<double>, py::arg("vals_batch"), py::arg("pat"),
          py::arg("num_threads") = -1,
          "Component sizes for each (W, W) float64 array of a batch (N, W, W), in parallel.");
    m.def("components_batch_signs", &sizes_batch<uint8_t>, py::arg("signs"), py::arg("pat"),
          py::arg("num_threads") = -1,
          "Component sizes for each (W, W) uint8 sign array (0/1) of a batch (N, W, W), in parallel.");
    m.def("count_components_batch", &count_components_batch, py::arg("signs"), py::arg("pat"),
          py::arg("min_size") = 1, py::arg("num_threads") = -1,
          "Number of components with at least min_size pixels for each uint8 sign array (0/1) of a batch "
          "(N, W, W), in parallel; returns an int32 array of length N.");
    m.def("count_bounded_batch", &count_bounded_batch, py::arg("signs"), py::arg("min_size") = 1,
          py::arg("num_threads") = -1, py::arg("eight") = 1,
          "Number of components (>= min_size pixels) not touching the border, for each (H, W) uint8 sign "
          "image of a batch (N, H, W); class `eight` is 8-connected, the other class 4-connected.");
}
