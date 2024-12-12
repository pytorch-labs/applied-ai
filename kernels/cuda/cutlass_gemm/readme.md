# CUTLASS FP8 GEMM

This project uses NVIDIA's CUTLASS library with Ping-Pong kernel on Hopper architecture design for efficient GPU-based GEMM.  [learn more](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
## Installation

- Prerequisites: NVIDIA Hopper GPU with CUDA support

### Without Docker
```bash
# 1. Clone the CUTLASS repository
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout 06b21349bcf6ddf6a1686a47a137ad1446579db9

# 2. Build CUTLASS
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON

# 3. Install the Python package
pip install -e .

# 4. Run the test script
python test_cutlass_gemm.py
```

### With Docker
```bash
# 1. Build the Docker image
docker build -t cutlass_gemm .

# 2. Run the Docker container
docker run --gpus all --rm -ti --ipc=host --name gpu_cutlass_gemm_instance cutlass_gemm /bin/bash

# 3. Inside the container, run the test script
python test_cutlass_gemm.py
```

