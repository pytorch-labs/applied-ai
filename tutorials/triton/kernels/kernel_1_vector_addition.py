import torch
import triton

import triton.language as tl  # gives access to the Triton language components
# see https://github.com/triton-lang/triton/blob/main/python/triton/language/__init__.py for all tl components

# we need to create a min of two components: the kernel and the kernel wrapper that both launches the kernel and interfaces with PyTorch
# the kernel is the actual Triton program that we want to run on the GPU

# @triton.jit is a decorator that designates a function as a Triton kernel
@triton.jit
def vector_addition_kernel(

    # the first two arguments are pointers to the input and output tensors
    # we will see how to pass these arguments in the wrapper function below
    x_ptr,
    y_ptr,
    output_ptr,

    # the total count of the inputs we are summing over
    n_elements,

    # the stride of the input and output tensors
    # tl.constexpr is a decorator that designates a variable as a compile-time constant
    # this allows Triton to perform compile-time optimizations b/c it knows
    # (and you guarantee via this decorator) the exact value of the variable at compile time
    # allowing the compiler to optimize for that value

    x_stride: tl.constexpr,
    y_stride: tl.constexpr,
    output_stride: tl.constexpr,

    # the block size for the grid and the multiplier in mapping of program ID to offsets
    BLOCK_SIZE: tl.constexpr,

):
    # the program ID is the unique identifier for each threadblock (note block!) in the grid
    # each PID will handle 'BLOCK_SIZE' elements
    pid = tl.program_id(axis=0)  # the grid is 1D so axis=0 means we move along the x-axis or columns in this case.
    block_start = pid * BLOCK_SIZE  # example: assuming pid==2, BLOCK_SIZE==128, then our current program will start at block_start==256

    # tl.arange(0, BLOCK_SIZE) will give us a range of 0 to 127, added to the block_start,
     # so offsets will be 256 to 383
     # note: tl.arange only generates a range of numbers that are power of 2 sizes, so we need to use the mask to
     # handle cases where the input is not a power of 2 size (i.e. 257 elements)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # the offsets for the input tensors
    x_offsets = offsets * x_stride  # stride is 1, so x_offsets will be 256 to 383
    y_offsets = offsets * y_stride  # stride is 1, so y_offsets will be 256 to 383

    # the mask for elements outside the range of [0, n_elements)
    mask = offsets < n_elements  # we need to handle cases where the input is not a power of 2 dimension (i.e. 257 elements)

    # load the input tensors
    x = tl.load(x_ptr + x_offsets, mask=mask)
    y = tl.load(y_ptr + y_offsets, mask=mask)

    # compute the output
    output = x + y

    # store the output
    tl.store(output_ptr + x_offsets, output, mask=mask) # stride is 1


# the kernel wrapper is a regular Python function that interfaces with PyTorch
# it is responsible for verifying the inputs, creating output buffers for results, and launching the kernel
# with appropriate grid size and information such as input strides
def vector_addition(x, y):

    # lets' first make sure that the input tensors are on the GPU
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU!"
    # we need to also make sure that a and b are the same size
    assert x.numel() == y.numel(), "Input tensors must be the same size!"


    # the output shape is the same as the input shape
    output = torch.empty_like(x)

    # the grid is the number of program blocks we need to launch to handle the entire input
    # cdiv just means ceiling division, so it returns the smallest integer >= x / y ensuring we always
    # launch enough program blocks to handle the entire input
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)

    # launch the kernel, note the grid is a lambda function that takes in the meta-arguments
    vector_addition_kernel[grid](
        # pass the pointers to the input and output tensors
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,

        # pass the total count of the input we are summing
        n_elements=x.numel(),

        # pass the strides (how many data element size jumps to get to the next element) of the input and output tensors
        # in this case, everything is contiguous, so we pass 1
        x_stride=1,
        y_stride=1,
        output_stride=1,

        # pass the block size for the grid and the mapping of program ID to offsets
        BLOCK_SIZE=128,

    )

    # return the output tensor
    return output


if __name__ == "__main__":
    # two tests - one for power of 2 size and one for non power of 2 size
    # the non power of 2 size is to test the mask functionality
    # create a random, power of 2 size, input tensor

    # Test 1: power of 2 size (1024!)

    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')

    # we can then use the kernel wrapper to perform a vector addition

    output = vector_addition(x, y)

    # verify the result with PyTorch reference implementation
    output_ref = x + y
    assert torch.allclose(output, output_ref)
    print("Success with power of 2 size (1024!")

    print(f"{output=}")

    # Test 2: non power of 2 size (257!)
    # create a non power of 2 input tensor
    x = torch.randn(257, device='cuda')
    y = torch.randn(257, device='cuda')

    # we can now use the kernel wrapper to perform a vector addition

    output_np2 = vector_addition(x, y)

    # verify the result with PyTorch reference implementation
    output_ref_np2 = x + y
    assert torch.allclose(output_np2, output_ref_np2)
    print("Success with non power of 2 size (num_elems = 257!)")

    print(f"{output_np2[0:5]=}")
