#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

void run_fht(void* a, void* out, uint32_t numel, uint32_t had_size, cudaStream_t stream);

torch::Tensor hadamard_transform(at::Tensor& in, bool inplace) {
    auto dtype = in.scalar_type();
    TORCH_CHECK(dtype == torch::ScalarType::Half, "Only fp16 supported currently");
    TORCH_CHECK(in.is_cuda());
    
    const int had_size = in.size(-1);
    TORCH_CHECK(had_size == 2 || had_size == 4 || had_size == 8 || had_size == 16
        || had_size == 32 || had_size == 64 || had_size == 128 || had_size == 256
        || had_size == 512 || had_size == 1024 || had_size == 2048 || had_size == 4096
        || had_size == 8192 || had_size == 16384 || had_size == 32768,
        "Only power of two Hadamard sizes up to 2^15 are supported, got ", had_size); // TODO: probably better way to do this
    torch::Tensor x = in.reshape({-1, had_size});
    
    auto numel = in.numel();
    if (numel % 256 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 0, 0, (numel % 256) / had_size}));
    }

    const auto res_shape = x.sizes();
    if (x.stride(-1) != 1)
        x = x.contiguous();
    torch::Tensor out = inplace ? x : torch::empty_like(x);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_fht(x.data_ptr(), out.data_ptr(), x.numel(), had_size, stream);

    if (inplace && out.data_ptr() != in.data_ptr()) {
        in.copy_(out.view(res_shape));
        out = in;
    }
    return out;
}

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard_transform", &hadamard_transform, "A function to perform a fast Hadamard transform", py::arg("x"), py::arg("inplace")=false);
}