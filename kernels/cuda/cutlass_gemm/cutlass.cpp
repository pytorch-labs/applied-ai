#include <torch/extension.h>
#include<torch/all.h>

void cutlass_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
                            torch::Tensor const& a_scales, torch::Tensor const& b_scales);

torch::Tensor cutlass_scaled_mm(torch::Tensor a, torch::Tensor b, torch::Tensor a_scales, torch::Tensor b_scales) {
    
    auto acc_dtype = torch::kFloat16;
    auto options = torch::TensorOptions().dtype(acc_dtype).device(a.device());
    torch::Tensor out = torch::empty({a.size(0), b.size(1)}, options);

    cutlass_scaled_mm_sm90(out, a, b, a_scales, b_scales);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_scaled_mm", &cutlass_scaled_mm, "CUTLASS Scaled Matrix Multiplication");
}