import logging

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import the grouped GEMM modules
from tgrouped_gemm_backwards import grouped_gemm_backward
from tgrouped_gemm_forward import grouped_gemm_forward as grouped_gemm


def test_backward_pass():
    """
    A simple test for the grouped GEMM backward pass with detailed error handling.
    Tests with small dimensions to focus on functionality.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test parameters
        G = 2  # Number of groups
        M = 256  # Input dimension
        N = 128  # Output dimension per group
        K = 128  # Hidden dimension

        # Create input and weight tensors
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
        w = torch.randn(
            N * G, K, dtype=torch.bfloat16, device=device, requires_grad=True
        )

        # Create group sizes
        m_sizes = torch.zeros(G, device=device, dtype=torch.int32)
        base_size = M // G
        remainder = M % G

        for i in range(G):
            m_sizes[i] = base_size + (1 if i < remainder else 0)

        # Log the setup
        print(f"Test setup - G: {G}, M: {M}, N: {N}, K: {K}")
        print(f"Input x shape: {x.shape}")
        logging.info(f"Weight w shape: {w.shape}")
        logging.info(f"Group sizes: {m_sizes}")

        # Step 1: Run forward pass
        logging.info("Running forward pass")
        result = grouped_gemm(x, w, m_sizes)
        logging.info(f"Forward result shape: {result.shape}")

        # Create a gradient for backpropagation
        grad_output = torch.randn_like(result)
        logging.info(f"Created gradient with shape: {grad_output.shape}")

        # Step 2: Run backward pass directly
        logging.info("Running backward pass directly")
        grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)

        # Verify gradient shapes
        logging.info(
            f"Gradient shapes - grad_x: {grad_x.shape}, grad_w: {grad_w.shape}"
        )

        # Step 3: Verify gradient computation using PyTorch's autograd
        # First create autograd-enabled tensors
        x_autograd = x.detach().clone().requires_grad_(True)
        w_autograd = w.detach().clone().requires_grad_(True)

        # Create a PyTorch reference implementation to compare against
        logging.info("Running PyTorch reference implementation")

        # Compute reference result
        reference_result = torch.zeros_like(result)
        m_start = 0
        for g in range(G):
            m_size = m_sizes[g].item()
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            if m_size > 0:
                reference_result[m_start:m_end, n_start:n_end] = (
                    x_autograd[m_start:m_end, :] @ w_autograd[n_start:n_end, :].T
                )

            m_start = m_end

        # Backpropagate using PyTorch
        reference_result.backward(grad_output)

        # Compare gradients
        logging.info("Comparing gradients with PyTorch reference")
        grad_x_error = (grad_x - x_autograd.grad).abs().max().item()
        grad_w_error = (grad_w - w_autograd.grad).abs().max().item()

        logging.info(f"grad W compare: {grad_w=}, {w_autograd=}")
        logging.info(f"grad X compare: {grad_x=}, {x_autograd=}")

        logging.info(
            f"Maximum gradient error - grad_x: {grad_x_error}, grad_w: {grad_w_error}"
        )

        # Check if gradients are close enough
        tolerance = 1e-2  # Higher tolerance for bfloat16
        if grad_x_error <= tolerance and grad_w_error <= tolerance:
            logging.info("✓ Gradients match the PyTorch reference")
        else:
            logging.error("✗ Gradient mismatch above tolerance threshold")

        return True

    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("Running test_backward_pass")
    logging.debug("Running test_backward_pass")
    success = test_backward_pass()
    logging.info(f"Test {'succeeded' if success else 'failed'}")
