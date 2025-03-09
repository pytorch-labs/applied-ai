import logging

import torch

""" Current results:
2025-03-09 15:03:09,494 - INFO -
--- Testing G=1, M=64, N=256, K=256 ---
2025-03-09 15:03:09,835 - INFO - Input x shape: torch.Size([64, 256])
2025-03-09 15:03:09,835 - INFO - Weight w shape: torch.Size([256, 256])
2025-03-09 15:03:09,836 - INFO - Group sizes m_sizes: tensor([64], device='cuda:0', dtype=torch.int32)
2025-03-09 15:03:15,574 - INFO - Forward result shape: torch.Size([64, 256])
2025-03-09 15:03:15,574 - INFO - Expected shape: torch.Size([64, 256])
2025-03-09 15:03:15,574 - INFO - ✓ Shape matches expected
2025-03-09 15:03:15,574 - INFO -
--- Testing G=4, M=64, N=256, K=256 ---
2025-03-09 15:03:15,575 - INFO - Input x shape: torch.Size([64, 256])
2025-03-09 15:03:15,575 - INFO - Weight w shape: torch.Size([1024, 256])
2025-03-09 15:03:15,575 - INFO - Group sizes m_sizes: tensor([16, 16, 16, 16], device='cuda:0', dtype=torch.int32)
2025-03-09 15:03:18,847 - INFO - Forward result shape: torch.Size([64, 1024])
2025-03-09 15:03:18,847 - INFO - Expected shape: torch.Size([64, 1024])
2025-03-09 15:03:18,847 - INFO - ✓ Shape matches expected
2025-03-09 15:03:18,847 - INFO -
--- Testing G=4, M=128, N=64, K=64 ---
2025-03-09 15:03:18,847 - INFO - Input x shape: torch.Size([128, 64])
2025-03-09 15:03:18,847 - INFO - Weight w shape: torch.Size([256, 64])
2025-03-09 15:03:18,848 - INFO - Group sizes m_sizes: tensor([32, 32, 32, 32], device='cuda:0', dtype=torch.int32)
2025-03-09 15:03:20,219 - INFO - Forward result shape: torch.Size([128, 256])
2025-03-09 15:03:20,219 - INFO - Expected shape: torch.Size([128, 256])
2025-03-09 15:03:20,219 - INFO - ✓ Shape matches expected
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import the grouped GEMM module
from tgrouped_gemm_forward import grouped_gemm_forward as grouped_gemm


def debug_forward_pass():
    """
    A simple test to debug the forward pass shape issues.
    Tests different configurations to identify where shape mismatches occur.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test case configurations to match the failing tests
    test_configs = [
        # G, M, N, K - these match the failing test cases
        (1, 64, 256, 256),  # First failing test
        (4, 64, 256, 256),  # Second failing test
        (4, 128, 64, 64),  # Custom groups test that's failing
    ]

    for G, M, N, K in test_configs:
        logging.info(f"\n--- Testing G={G}, M={M}, N={N}, K={K} ---")

        # Create input and weight tensors
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)

        # Create group sizes
        m_sizes = torch.zeros(G, device=device, dtype=torch.int32)
        base_size = M // G
        remainder = M % G

        for i in range(G):
            m_sizes[i] = base_size + (1 if i < remainder else 0)

        # Log inputs
        logging.info(f"Input x shape: {x.shape}")
        logging.info(f"Weight w shape: {w.shape}")
        logging.info(f"Group sizes m_sizes: {m_sizes}")

        # Run forward pass
        result = grouped_gemm(x, w, m_sizes)
        logging.info(f"Forward result shape: {result.shape}")

        # Calculate expected shape
        expected_shape = torch.Size([M, N * G])
        logging.info(f"Expected shape: {expected_shape}")

        # Check if shapes match
        if result.shape == expected_shape:
            logging.info("✓ Shape matches expected")
        else:
            logging.error(
                f"✗ Shape mismatch: got {result.shape}, expected {expected_shape}"
            )

            # Try to figure out what's happening
            # Check if the output is [M, N] instead of [M, N*G]
            if result.shape == torch.Size([M, N]):
                logging.error(
                    "The implementation seems to be using N instead of N*G for output width"
                )

            # Analyze the shape to understand the pattern
            ratio_w = result.shape[1] / expected_shape[1]
            logging.error(f"Output width ratio (actual/expected): {ratio_w}")

            # Compute reference result with PyTorch for comparison
            reference = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)
            m_start = 0
            for g in range(G):
                m_size = m_sizes[g].item()
                m_end = m_start + m_size
                n_start = g * N
                n_end = (g + 1) * N

                if m_size > 0:
                    chunk = x[m_start:m_end, :] @ w[n_start:n_end, :].T
                    logging.info(f"Group {g} chunk shape: {chunk.shape}")
                    reference[m_start:m_end, n_start:n_end] = chunk

                m_start = m_end

            logging.info(f"Reference result shape: {reference.shape}")


if __name__ == "__main__":
    debug_forward_pass()
