Applied AI repo 

# NanoGPT 2D (Tensor Parallel and FSDP)
This is the NanoGPT project from Andrej Karpathy, 
updated to use PyTorch Tensor Parallel and FSDP.

(Original repo = https://github.com/karpathy/nanoGPT)

To run:
a - set a reasonable model size in config/nanogpt_2D.py  (model_name = "1B" or similar from commented list)
b - update the file "run_tp.sh" to ensure you have matching gpu count to your server (nproc_per_node=your_gpu_count)
c - "bash run_tp.sh" to launch
d - You can toggle between using Tensor Parallel or only FSDP via config / nanogpt_2D.py - use_tensor_parallel: bool flag.

Note - the default dataset used is a subset of OpenWebText.  This is to ensure demo is 'ready to run'. 
It will run 8 iterations by default and showcase the avg iter time, gpu memory, mfu and other statistics. 


