#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include <mutex>

namespace py = pybind11;

// ------------------------------------------------------------
// Union–Find (Disjoint Set Union)
// ------------------------------------------------------------
struct DSU {
    std::vector<int> parent, size;

    DSU(int n) : parent(n), size(n, 1) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
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

// ------------------------------------------------------------
// Label connected components - single array
// ------------------------------------------------------------
py::list components(py::array_t<double> vals, py::array_t<int> pat) {
    auto vals_buf = vals.unchecked<2>();               // [width, width]
    auto pat_buf  = pat.unchecked<4>();                // [12, 2, width, width]

    int width = vals_buf.shape(0);

    // labels[i][j] = group index (initially -1)
    std::vector<std::vector<int>> labels(width, std::vector<int>(width, -1));

    // Maximum possible groups = width*width
    DSU dsu(width * width);
    int next_group = 0;

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {

            double v = vals_buf(i, j);

            int chosen = -1;                        // best existing label
            std::vector<int> neighbors;             // compatible neighbors

            neighbors.reserve(12);

            for (int k = 0; k < 12; ++k) {
                int d0 = pat_buf(k, 0, i, j);
                int d1 = pat_buf(k, 1, i, j);
                double vn = vals_buf(d0, d1);
                if ((v >= 0 && vn >= 0) || (v < 0 && vn < 0)) {
                    int L = labels[d0][d1];
                    if (L != -1) {
                        neighbors.push_back(L);
                        if (L > chosen) chosen = L;
                    }
                }
            }

            // No compatible labeled neighbor → create new group
            if (chosen == -1) {
                labels[i][j] = next_group++;
                continue;
            }

            // Attach to chosen group
            labels[i][j] = chosen;
            int root0 = dsu.find(chosen);
            dsu.size[root0] += 1;

            // If fewer than 2 distinct neighbors → no merging
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            if ((int)neighbors.size() < 2) continue;

            // Merge all these groups through Union-Find
            for (int g : neighbors)
                dsu.unite(root0, g);
        }
    }

    // ------------------------------------------------------------
    // Compute group sizes
    // ------------------------------------------------------------
    std::vector<int> roots;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            int root = dsu.find(labels[i][j]);
            roots.push_back(root);
        }
    }
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

    // Return Python list of sizes of the connected components
    py::list out;
    for (auto v : roots) out.append(dsu.size[v]);
    return out;
}

// -------------------- Single-array computation (C++ data pointers) --------------------
// vals_ptr points to contiguous double array of size W*W for a single item
// pat_ptr points to contiguous int array of size 12*2*W*W (layout described below)
// Fills labels (size W*W) with group ids; dsu.find(label) gives the component root.
static void
label_single_array(int W, const double *vals_ptr, const int *pat_ptr, std::vector<int> &labels, DSU &dsu) {
    int next_group = 0;

    auto idx = [W](int i, int j) { return i * W + j; };

    // pat layout: pat_ptr[k*2*W*W + p*W*W + i*W + j]
    auto pat_index = [W](int k, int p, int i, int j) {
        return k * 2 * W * W + p * W * W + i * W + j;
    };

    for (int i = 0; i < W; ++i) {
        for (int j = 0; j < W; ++j) {
            int pos = idx(i, j);
            double v = vals_ptr[pos];

            int chosen = -1;
            std::vector<int> neighbors; neighbors.reserve(12);

            for (int k = 0; k < 12; ++k) {
                int d0 = pat_ptr[ pat_index(k, 0, i, j) ];
                int d1 = pat_ptr[ pat_index(k, 1, i, j) ];
                int dpos = idx(d0, d1);
                double other = vals_ptr[dpos];
                if ((v < 0 && other < 0) || (v >= 0 && other >= 0)) {
                    int L = labels[dpos];
                    if (L != -1) {
                        neighbors.push_back(L);
                        if (L > chosen) chosen = L;
                    }
                }
            }

            if (chosen == -1) {
                // new group id
                labels[pos] = next_group++;
                // DSU already sized to Npos; group ids are in [0, next_group-1]
            } else {
                labels[pos] = chosen;
                int root0 = dsu.find(chosen);
                dsu.size[root0] += 1;
                
                // we only care about distinct neighbor labels
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

                if ((int)neighbors.size() < 2) continue;
                // merge all neighbor groups with the chosen group
                
                for (int g : neighbors)
                    dsu.unite(root0, g);
            }
        }
    }

}

// Sorted list of distinct component roots
static std::vector<int> component_roots(const std::vector<int> &labels, DSU &dsu) {
    std::vector<int> roots;
    roots.reserve(labels.size());
    for (int L : labels) roots.push_back(dsu.find(L));
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    return roots;
}

// Sizes of the connected components, in the order of their sorted roots
static std::vector<int>
compute_single_array(int W, const double *vals_ptr, const int *pat_ptr) {
    std::vector<int> labels(W * W, -1);
    DSU dsu(W * W);
    label_single_array(W, vals_ptr, pat_ptr, labels, dsu);
    std::vector<int> out;
    for (int r : component_roots(labels, dsu)) out.push_back(dsu.size[r]);
    return out;
}

// ------------------------------------------------------------
// Component labels of a single array: returns (labels[W, W], sizes), where
// labels[i, j] is the index of the component of pixel (i, j) in sizes
// (same order as the output of components / components_batch).
// ------------------------------------------------------------
py::tuple component_labels(py::array_t<double, py::array::c_style | py::array::forcecast> vals,
                           py::array_t<int, py::array::c_style | py::array::forcecast> pat) {
    py::buffer_info vb = vals.request();
    py::buffer_info pb = pat.request();
    if (vb.ndim != 2 || vb.shape[0] != vb.shape[1]) throw std::runtime_error("vals must be a 2D array (W, W)");
    const int W = (int) vb.shape[0];
    if (pb.ndim != 4 || (int)pb.shape[0] != 12 || (int)pb.shape[1] != 2 || (int)pb.shape[2] != W || (int)pb.shape[3] != W)
        throw std::runtime_error("pat must have shape (12,2,W,W) with same W as vals");

    std::vector<int> labels(W * W, -1);
    DSU dsu(W * W);
    label_single_array(W, static_cast<const double*>(vb.ptr), static_cast<const int*>(pb.ptr), labels, dsu);
    std::vector<int> roots = component_roots(labels, dsu);

    py::array_t<int> out_labels({W, W});
    auto ol = out_labels.mutable_unchecked<2>();
    for (int i = 0; i < W; ++i)
        for (int j = 0; j < W; ++j) {
            int r = dsu.find(labels[i * W + j]);
            ol(i, j) = (int)(std::lower_bound(roots.begin(), roots.end(), r) - roots.begin());
        }
    py::list sizes;
    for (int r : roots) sizes.append(dsu.size[r]);
    return py::make_tuple(out_labels, sizes);
}

// -------------------- Batch parallel wrapper --------------------
py::list components_batch(py::array_t<double, py::array::c_style | py::array::forcecast> vals_batch,
                          py::array_t<int, py::array::c_style | py::array::forcecast> pat,
                          int num_threads = -1) {
    // Validate inputs and extract raw pointers and shapes while GIL is held
    py::buffer_info vb = vals_batch.request();
    py::buffer_info pb = pat.request();

    if (vb.ndim != 3) throw std::runtime_error("vals_batch must be a 3D array (N, W, W)");
    if (pb.ndim != 4) throw std::runtime_error("pat must be a 4D array (12, 2, W, W)");

    const int N = (int) vb.shape[0];
    const int W0 = (int) vb.shape[1];
    const int W1 = (int) vb.shape[2];
    if (W0 != W1) throw std::runtime_error("vals_batch second and third dims must be equal (W,W)");
    const int W = W0;

    if ((int)pb.shape[0] != 12 || (int)pb.shape[1] != 2 || (int)pb.shape[2] != W || (int)pb.shape[3] != W)
        throw std::runtime_error("pat must have shape (12,2,W,W) with same W as vals_batch");

    // Pointers to raw data (contiguous C-order)
    double *vals_data = static_cast<double*>(vb.ptr);
    int *pat_data  = static_cast<int*>(pb.ptr);

    // Per-item results stored as C++ maps (one per array)
    std::vector<std::vector<int>> results;
    results.resize(N);

    if (N == 0) return py::list();

    // Determine number of threads
    unsigned int hw = std::thread::hardware_concurrency();
    int tcount = num_threads <= 0 ? (hw ? (int)hw : 1) : num_threads;
    if (tcount < 1) tcount = 1;
    if (tcount > N) tcount = N;

    // Partition indices into tcount ranges
    std::vector<int> start_idx(tcount), end_idx(tcount);
    int base = N / tcount;
    int rem = N % tcount;
    int cur = 0;
    for (int t = 0; t < tcount; ++t) {
        start_idx[t] = cur;
        int add = base + (t < rem ? 1 : 0);
        cur += add;
        end_idx[t] = cur; // [start_idx[t], end_idx[t])
    }

    // Release GIL for heavy computation
    {
        py::gil_scoped_release release;

        std::vector<std::thread> threads;
        threads.reserve(tcount);

        for (int t = 0; t < tcount; ++t) {
            int s = start_idx[t], e = end_idx[t];
            threads.emplace_back([s,e,W,N,vals_data,pat_data,&results](void) {
                const size_t single_size = (size_t)W * W;
                for (int idx = s; idx < e; ++idx) {
                    // pointer to this array's data: offset = idx * W * W
                    const double *vals_ptr = vals_data + (size_t)idx * single_size;
                    // pat is the same pointer for all arrays
                    auto res = compute_single_array(W, vals_ptr, pat_data);
                    results[idx] = std::move(res);
                }
            });
        }

        for (auto &th : threads) if (th.joinable()) th.join();
    } // GIL reacquired here

    // Convert results (vector of unordered_map) to Python list of dicts
    py::list out;
    for (int i = 0; i < N; ++i) {
        py::list d;
        for (auto v : results[i]) d.append(v);
        out.append(d);
    }
    return out;
}

// ------------------------------------------------------------
// Pybind11 module definition
// ------------------------------------------------------------
PYBIND11_MODULE(_connected, m) {
    m.doc() = "Optimized connected components formation using Union-Find (pybind11)";
    m.def("components", &components, py::arg("vals"), py::arg("pat"));
    m.def("component_labels", &component_labels, py::arg("vals"), py::arg("pat"),
          "Component labels of a single array: returns (labels[W, W], sizes).");
    m.def("components_batch", &components_batch,
          py::arg("vals_batch"), py::arg("pat"), py::arg("num_threads") = -1,
          "Find connected components for each 2D array in vals_batch (shape N x W x W) in parallel.\n"
          "pat must have shape (12,2,W,W). Returns a list of lists (one list per input array"
          "containing the sizes of the components).");
}
