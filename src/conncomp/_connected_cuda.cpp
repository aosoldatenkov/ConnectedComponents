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

static void label_faces(std::uintptr_t signs, std::uintptr_t parent, std::uintptr_t seam_pairs, int n_seams, int B,
                        int M, int eight, std::uintptr_t stream) {
    if (6LL * M * M * B >= (1LL << 31)) throw std::runtime_error("batch too large for int32 node indices");
    const int err = ccl_label_faces(reinterpret_cast<const uint8_t *>(signs), reinterpret_cast<int *>(parent),
                                    reinterpret_cast<const int *>(seam_pairs), n_seams, B, M, eight,
                                    reinterpret_cast<cudaStream_t>(stream));
    if (err != 0) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString((cudaError_t)err));
}

PYBIND11_MODULE(_connected_cuda, m) {
    m.doc() = "Block-based union-find labelling on the six cube-face grids (CUDA)";
    m.def("label_faces", &label_faces, py::arg("signs"), py::arg("parent"), py::arg("seam_pairs"),
          py::arg("n_seams"), py::arg("B"), py::arg("M"), py::arg("eight"), py::arg("stream"),
          "Component labels (root node index per node) for uint8 signs (B, 6, M, M) on the device; all arguments "
          "are device pointers / sizes / a cudaStream_t handle.");
}
