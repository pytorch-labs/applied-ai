Applied AI repo
Housing a variety of Triton and CUDA kernels for training and inference.
Inference kernels = no backward pass support.

Newest additions:
1 - Triton - MoE (Mixtral) GEMM for accelerating inference. Uses col major weight layout to increase locality. 

<img width="556" alt="Screenshot 2024-03-18 at 5 10 58 PM" src="https://github.com/lessw2020/applied-ai/assets/46302957/7edffa8c-601e-485c-bbc8-64b734ee8ced">




2 - Triton - Fused Softmax for both training and inference. 

<img width="556" alt="fused_softmax_a100" src="https://github.com/lessw2020/applied-ai/assets/46302957/4f2daefc-0ea3-4ee6-b9fe-181382fb518b">

