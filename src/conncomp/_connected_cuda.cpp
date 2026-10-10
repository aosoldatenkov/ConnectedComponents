// Python bindings for the CUDA connected-component labelling (_cuda_ccl.cu). Device memory is passed as raw
// pointers (e.g. torch.Tensor.data_ptr()) together with the CUDA stream handle, so the module does not depend on
// PyTorch's C++ API.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace py = pybind11;

int ccl_label_faces(const uint8_t *signs, int *parent, const int *seam_pairs, int n_seams, int B, int M, int eight,
                    cudaStream_t stream);

int ccl_form_trees(const uint8_t *signs, const int *labels, const int *weight, const int *anti, int B, int M,
                   int min_size, int *nreg, int *nov, int *ok, int *size, int *sign, int *flags, int *parent, int *depth,
                   cudaStream_t stream);
int ccl_tree_capacity();

static void check(int err) {
    if (err != 0) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString((cudaError_t)err));
}

static void form_trees(std::uintptr_t signs, std::uintptr_t labels, std::uintptr_t weight, std::uintptr_t anti, int B,
                       int M, int min_size, std::uintptr_t nreg, std::uintptr_t nov, std::uintptr_t ok,
                       std::uintptr_t size, std::uintptr_t sign, std::uintptr_t flags, std::uintptr_t parent,
                       std::uintptr_t depth, std::uintptr_t stream) {
    if (6LL * M * M * B >= (1LL << 31)) throw std::runtime_error("batch too large for int32 node indices");
    auto I = [](std::uintptr_t p) { return reinterpret_cast<int *>(p); };
    check(ccl_form_trees(reinterpret_cast<const uint8_t *>(signs), I(labels), I(weight), I(anti), B, M, min_size,
                         I(nreg), I(nov), I(ok), I(size), I(sign), I(flags), I(parent), I(depth),
                         reinterpret_cast<cudaStream_t>(stream)));
}

static void label_faces(std::uintptr_t signs, std::uintptr_t parent, std::uintptr_t seam_pairs, int n_seams, int B,
                        int M, int eight, std::uintptr_t stream) {
    if (6LL * M * M * B >= (1LL << 31)) throw std::runtime_error("batch too large for int32 node indices");
    const int err = ccl_label_faces(reinterpret_cast<const uint8_t *>(signs), reinterpret_cast<int *>(parent),
                                    reinterpret_cast<const int *>(seam_pairs), n_seams, B, M, eight,
                                    reinterpret_cast<cudaStream_t>(stream));
    check(err);
}

PYBIND11_MODULE(_connected_cuda, m) {
    m.doc() = "Block-based union-find labelling on the six cube-face grids (CUDA)";
    m.def("label_faces", &label_faces, py::arg("signs"), py::arg("parent"), py::arg("seam_pairs"),
          py::arg("n_seams"), py::arg("B"), py::arg("M"), py::arg("eight"), py::arg("stream"),
          "Component labels (root node index per node) for uint8 signs (B, 6, M, M) on the device; all arguments "
          "are device pointers / sizes / a cudaStream_t handle.");
    m.def("form_trees", &form_trees, py::arg("signs"), py::arg("labels"), py::arg("weight"), py::arg("anti"),
          py::arg("B"), py::arg("M"), py::arg("min_size"), py::arg("nreg"), py::arg("nov"), py::arg("ok"),
          py::arg("size"), py::arg("sign"), py::arg("flags"), py::arg("parent"), py::arg("depth"), py::arg("stream"),
          "Region trees from labels, one block per form. Outputs (int32 device arrays): nreg, nov, ok (B,) and size, "
          "sign, flags (bit 0 alive, bit 1 root), parent, depth (B, capacity); nreg = -1 marks forms with more than "
          "`capacity` components.");
    m.attr("tree_capacity") = ccl_tree_capacity();
}
