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
// Region trees (nesting of ovals) via the double cover S^2 -> RP^2
// ------------------------------------------------------------
// Nodes are (pixel, sheet): the 8 adjacent neighbours keep the sheet, the 4 antipodal neighbours switch it, so the
// graph models S^2. The two lifts of a region of RP^2 are a component and its sheet-swapped copy; the region is
// non-orientable (the outside region N of an even-degree curve) iff its two lifts coincide. Regions are vertices,
// pairs of adjacent regions (opposite signs) are edges (ovals); for a smooth even-degree curve this is a tree.
struct RegionTree {
    std::vector<int> size, sign, parent, depth, self;  // per region (after pruning); parent of the root = -1
    int root = -1, n_roots = 0;                  // number of self-lifting regions before pruning (1 expected)
    bool tree = false;                           // edges == regions - 1 and connected
};

static RegionTree build_tree(int R, const std::vector<int> &size, const std::vector<int> &sign,
                             const std::vector<char> &is_self, std::vector<long long> pairs, int min_size) {
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    std::vector<std::vector<int>> adj(R);
    for (long long e : pairs) {
        const int a = (int)(e / R), b = (int)(e % R);
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    RegionTree t;
    for (int r = 0; r < R; ++r) t.n_roots += is_self[r];
    // prune small leaf regions (spurious ovals), repeatedly
    std::vector<char> alive(R, 1);
    std::vector<int> deg(R);
    for (int r = 0; r < R; ++r) deg[r] = (int)adj[r].size();
    bool changed = true;
    while (changed) {
        changed = false;
        for (int r = 0; r < R; ++r)
            if (alive[r] && !is_self[r] && size[r] < min_size && deg[r] <= 1) {
                alive[r] = 0;
                changed = true;
                for (int q : adj[r])
                    if (alive[q]) deg[q]--;
            }
    }
    int root = -1;
    for (int r = 0; r < R; ++r)
        if (alive[r] && is_self[r]) root = r;
    std::vector<int> newid(R, -1);
    int n = 0;
    for (int r = 0; r < R; ++r)
        if (alive[r]) newid[r] = n++;
    t.size.resize(n);
    t.sign.resize(n);
    t.self.resize(n);
    t.parent.assign(n, -2);
    t.depth.assign(n, -1);
    int edges = 0;
    for (int r = 0; r < R; ++r) {
        if (!alive[r]) continue;
        t.size[newid[r]] = size[r];
        t.sign[newid[r]] = sign[r];
        t.self[newid[r]] = is_self[r];
        for (int q : adj[r])
            if (alive[q] && q > r) edges++;
    }
    if (root >= 0) {  // breadth-first search from the root
        t.root = newid[root];
        std::vector<int> queue{root};
        t.parent[newid[root]] = -1;
        t.depth[newid[root]] = 0;
        for (size_t h = 0; h < queue.size(); ++h) {
            const int r = queue[h];
            for (int q : adj[r])
                if (alive[q] && t.depth[newid[q]] < 0) {
                    t.parent[newid[q]] = newid[r];
                    t.depth[newid[q]] = t.depth[newid[r]] + 1;
                    queue.push_back(q);
                }
        }
        t.tree = (edges == n - 1) && ((int)queue.size() == n);
    }
    return t;
}

static RegionTree region_tree_single(int W, const uint8_t *s, const int *pat, DSU &dsu, int min_size) {
    const int P = W * W;
    for (int p = 0; p < P; ++p) {
        const int c = s[p] != 0;
        for (int k = 0; k < 12; ++k) {
            const int q = pat[k * 2 * P + p] * W + pat[k * 2 * P + P + p];
            if (q == p || (s[q] != 0) != c) continue;
            if (k < 8) {
                if (q < p) continue;  // symmetric adjacency: once per edge
                dsu.unite(p, q);
                dsu.unite(P + p, P + q);
            } else {
                dsu.unite(p, P + q);
                dsu.unite(P + p, q);
            }
        }
    }
    // region key per pixel and compact region indices
    std::vector<int> key(P);
    std::vector<char> self(P);
    for (int p = 0; p < P; ++p) {
        const int a = dsu.find(p), b = dsu.find(P + p);
        key[p] = std::min(a, b);
        self[p] = (a == b);
    }
    std::vector<int> keys(key);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const int R = (int)keys.size();
    auto idx = [&](int k) { return (int)(std::lower_bound(keys.begin(), keys.end(), k) - keys.begin()); };
    std::vector<int> reg(P), size(R, 0), sign(R, 0);
    std::vector<char> is_self(R, 0);
    for (int p = 0; p < P; ++p) {
        const int r = idx(key[p]);
        reg[p] = r;
        size[r]++;
        sign[r] = s[p] != 0 ? 1 : -1;
        if (self[p]) is_self[r] = 1;
    }
    // adjacency between regions (pairs of neighbouring pixels of opposite sign)
    std::vector<long long> pairs;
    for (int p = 0; p < P; ++p)
        for (int k = 0; k < 12; ++k) {
            const int q = pat[k * 2 * P + p] * W + pat[k * 2 * P + P + p];
            if (q == p || reg[q] == reg[p]) continue;
            const int a = std::min(reg[p], reg[q]), b = std::max(reg[p], reg[q]);
            pairs.push_back((long long)a * R + b);
        }
    return build_tree(R, size, sign, is_self, pairs, min_size);
}

static py::tuple pack_trees(const std::vector<RegionTree> &trees) {
    const int N = (int)trees.size();
    std::vector<int> offsets(N + 1, 0);
    for (int i = 0; i < N; ++i) offsets[i + 1] = offsets[i] + (int)trees[i].size.size();
    const int T = offsets[N];
    py::array_t<int> off(N + 1), size(T), sign(T), parent(T), depth(T), selfl(T), root(N), nroots(N), tree(N);
    int *o = off.mutable_data(), *sz = size.mutable_data(), *sg = sign.mutable_data(), *pa = parent.mutable_data(),
        *de = depth.mutable_data(), *sl = selfl.mutable_data(), *ro = root.mutable_data(), *nr = nroots.mutable_data(),
        *tr = tree.mutable_data();
    for (int i = 0; i <= N; ++i) o[i] = offsets[i];
    for (int i = 0; i < N; ++i) {
        const RegionTree &t = trees[i];
        std::copy(t.size.begin(), t.size.end(), sz + offsets[i]);
        std::copy(t.sign.begin(), t.sign.end(), sg + offsets[i]);
        std::copy(t.parent.begin(), t.parent.end(), pa + offsets[i]);
        std::copy(t.depth.begin(), t.depth.end(), de + offsets[i]);
        std::copy(t.self.begin(), t.self.end(), sl + offsets[i]);
        ro[i] = t.root;
        nr[i] = t.n_roots;
        tr[i] = t.tree ? 1 : 0;
    }
    return py::make_tuple(off, size, sign, parent, depth, root, nroots, tree, selfl);
}

// For each (W, W) uint8 sign image of a batch: the region tree. Returns (offsets (N + 1), size, sign, parent,
// depth (concatenated per region), root (N), n_roots (N), is_tree (N), self (per region: lift connected)); region
// indices are local to each form.
py::tuple region_trees_batch(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> signs,
                             py::array_t<int, py::array::c_style | py::array::forcecast> pat, int min_size,
                             int num_threads) {
    py::buffer_info sb = signs.request(), pb = pat.request();
    if (sb.ndim != 3 || sb.shape[1] != sb.shape[2]) throw std::runtime_error("signs must have shape (N, W, W)");
    const int N = (int)sb.shape[0], W = (int)sb.shape[1], P = W * W;
    check_pat(pb, W);
    const uint8_t *s = static_cast<const uint8_t *>(sb.ptr);
    const int *pp = static_cast<const int *>(pb.ptr);
    std::vector<RegionTree> trees(N);
    if (N > 0)
        parallel_for(N, 2 * P, num_threads, [&](int i, DSU &dsu) {
            trees[i] = region_tree_single(W, s + (size_t)i * P, pp, dsu, min_size);
        });
    return pack_trees(trees);
}

// ------------------------------------------------------------
// Region trees on a general mesh of S^2 with an antipodal map (e.g. the cube-surface lattice)
// ------------------------------------------------------------
// nbr4 (V, K4): neighbours used by both sign classes; nbr8 (V, K8): extra neighbours (e.g. square diagonals) used
// only by the class `eight` (consistent 8/4 digital topology). -1 pads. antipode (V): index of -p.
// Components on S^2 are computed directly; a region of RP^2 is a component together with its antipodal image,
// and the non-orientable region N is the component that is its own antipodal image.
static RegionTree region_tree_mesh_single(const uint8_t *s, const int *nbr4, int K4, const int *nbr8, int K8,
                                          const int *anti, int V, int eight, DSU &dsu, int min_size) {
    for (int p = 0; p < V; ++p) {
        const int c = s[p] != 0;
        for (int k = 0; k < K4; ++k) {
            const int q = nbr4[p * K4 + k];
            if (q > p && (s[q] != 0) == c) dsu.unite(p, q);
        }
        if (c == eight)
            for (int k = 0; k < K8; ++k) {
                const int q = nbr8[p * K8 + k];
                if (q > p && (s[q] != 0) == c) dsu.unite(p, q);
            }
    }
    std::vector<int> key(V);
    std::vector<char> self(V);
    for (int p = 0; p < V; ++p) {
        const int a = dsu.find(p), b = dsu.find(anti[p]);
        key[p] = std::min(a, b);
        self[p] = (a == b);
    }
    std::vector<int> keys(key);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const int R = (int)keys.size();
    std::vector<int> reg(V), size(R, 0), sign(R, 0);
    std::vector<char> is_self(R, 0);
    for (int p = 0; p < V; ++p) {
        const int r = (int)(std::lower_bound(keys.begin(), keys.end(), key[p]) - keys.begin());
        reg[p] = r;
        size[r]++;
        sign[r] = s[p] != 0 ? 1 : -1;
        if (self[p]) is_self[r] = 1;
    }
    std::vector<long long> pairs;
    for (int p = 0; p < V; ++p)
        for (int k = 0; k < K4; ++k) {
            const int q = nbr4[p * K4 + k];
            if (q < 0 || reg[q] == reg[p]) continue;
            const int a = std::min(reg[p], reg[q]), b = std::max(reg[p], reg[q]);
            pairs.push_back((long long)a * R + b);
        }
    return build_tree(R, size, sign, is_self, pairs, min_size);
}

py::tuple region_trees_mesh(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> signs,
                            py::array_t<int, py::array::c_style | py::array::forcecast> nbr4,
                            py::array_t<int, py::array::c_style | py::array::forcecast> nbr8,
                            py::array_t<int, py::array::c_style | py::array::forcecast> antipode, int eight,
                            int min_size, int num_threads) {
    py::buffer_info sb = signs.request(), b4 = nbr4.request(), b8 = nbr8.request(), ba = antipode.request();
    if (sb.ndim != 2) throw std::runtime_error("signs must have shape (N, V)");
    const int N = (int)sb.shape[0], V = (int)sb.shape[1];
    if (b4.ndim != 2 || b4.shape[0] != V || b8.ndim != 2 || b8.shape[0] != V || ba.ndim != 1 || ba.shape[0] != V)
        throw std::runtime_error("nbr4 (V, K4), nbr8 (V, K8) and antipode (V,) must match signs (N, V)");
    const int K4 = (int)b4.shape[1], K8 = (int)b8.shape[1];
    const uint8_t *s = static_cast<const uint8_t *>(sb.ptr);
    const int *n4 = static_cast<const int *>(b4.ptr), *n8 = static_cast<const int *>(b8.ptr);
    const int *an = static_cast<const int *>(ba.ptr);
    std::vector<RegionTree> trees(N);
    if (N > 0)
        parallel_for(N, V, num_threads, [&](int i, DSU &dsu) {
            trees[i] = region_tree_mesh_single(s + (size_t)i * V, n4, K4, n8, K8, an, V, eight, dsu, min_size);
        });
    return pack_trees(trees);
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
    m.def("region_trees_batch", &region_trees_batch, py::arg("signs"), py::arg("pat"), py::arg("min_size") = 3,
          py::arg("num_threads") = -1,
          "Region trees (nesting of ovals) for each (W, W) uint8 sign image of a batch, via the double cover; returns "
          "(offsets, size, sign, parent, depth, root, n_roots, is_tree).");
    m.def("region_trees_mesh", &region_trees_mesh, py::arg("signs"), py::arg("nbr4"), py::arg("nbr8"),
          py::arg("antipode"), py::arg("eight") = 1, py::arg("min_size") = 3, py::arg("num_threads") = -1,
          "Region trees for sign vectors (N, V) on a mesh of S^2 with neighbour tables and an antipodal map.");
    m.def("count_bounded_batch", &count_bounded_batch, py::arg("signs"), py::arg("min_size") = 1,
          py::arg("num_threads") = -1, py::arg("eight") = 1,
          "Number of components (>= min_size pixels) not touching the border, for each (H, W) uint8 sign "
          "image of a batch (N, H, W); class `eight` is 8-connected, the other class 4-connected.");
}
