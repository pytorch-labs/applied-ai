# To build the image, run the following command:
# docker build -t cutlass_gemm .
# To run the image, run the following command:
# docker run --gpus all --rm -ti --ipc=host --name gpu_cutlass_gemm_instance cutlass_gemm /bin/bash

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install common dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        sudo \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
COPY ./ /workspace
WORKDIR /workspace
ENV PYTHONPATH /workspace:$PYTHONPATH

# Clone the cutlass repository
RUN git clone https://github.com/NVIDIA/cutlass.git /workspace/cutlass
RUN cd /workspace/cutlass && git checkout 06b21349bcf6ddf6a1686a47a137ad1446579db9
# Install cutlass
RUN cd /workspace/cutlass && mkdir -p build
RUN cd /workspace/cutlass/build && cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON

# Install cutlass gemm
RUN cd /workspace/ && pip install -e .