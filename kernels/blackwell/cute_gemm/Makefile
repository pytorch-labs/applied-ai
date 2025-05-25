
# Makefile for SM100 GEMM PyTorch Extension

# Set these paths according to your installation
CUTLASS_PATH ?= /path/to/cutlass
CUDA_HOME ?= $(shell python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME)")

# Build the extension
build:
	CUTLASS_PATH=$(CUTLASS_PATH) python setup.py build_ext --inplace

# Install the extension
install:
	CUTLASS_PATH=$(CUTLASS_PATH) pip install .

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ sm100_gemm*.so

# Test the installation
test:
	python python_interface.py

# Check CUDA device capability
check_device:
	python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name()}, Compute capability: {torch.cuda.get_device_capability()}')"

.PHONY: build install clean test check_device
