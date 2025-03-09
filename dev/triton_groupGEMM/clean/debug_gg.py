import torch
import logging

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import the grouped GEMM modules
from groupgemm import grouped_gemm
from tgrouped_gemm_backwards import grouped_gemm_backward

def debug_grouped_gemm():
    """
    A simple function to debug the grouped GEMM forward and backward passes.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set parameters for a simple test
    G = 2  # Number of groups
    M = 32  # Input dimension
    N = 16  # Output dimension per group
    K = 16  # Hidden dimension
    
    # Create input and weight tensors
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    w = torch.randn(N * G, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    
    # Create group sizes
    m_sizes = torch.zeros(G, device=device, dtype=torch.int32)
    base_size = M // G
    remainder = M % G
    
    for i in range(G):
        m_sizes[i] = base_size + (1 if i < remainder else 0)
    
    # Print key information
    logging.info(f"Input x shape: {x.shape}")
    logging.info(f"Weight w shape: {w.shape}")
    logging.info(f"Group sizes m_sizes: {m_sizes}")
    logging.info(f"Sum of group sizes: {m_sizes.sum().item()}")
    
    # Inspect the forward function source code
    try:
        from inspect import getsource
        from groupgemm import _grouped_gemm
        logging.info(f"Forward function signature: {getsource(_grouped_gemm).splitlines()[0]}")
    except Exception as e:
        logging.error(f"Could not get source code: {e}")
    
    # Test forward pass
    try:
        logging.info("Testing forward pass...")
        result = grouped_gemm(x, w, m_sizes)
        logging.info(f"Forward pass output shape: {result.shape}")
        
        # Expected output shape calculation
        expected_shape = torch.Size([M, N * G])
        logging.info(f"Expected output shape: {expected_shape}")
        
        # Compute reference result with PyTorch operations
        expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)
        m_start = 0
        for g in range(G):
            m_size = m_sizes[g].item()
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N
            
            # Only compute for non-empty groups
            if m_size > 0:
                expected_result[m_start:m_end, n_start:n_end] = (
                    x[m_start:m_end, :] @ w[n_start:n_end, :].T
                )
            
            m_start = m_end
        
        # Check shapes match
        shape_match = result.shape == expected_result.shape
        logging.info(f"Shape match: {shape_match}")
        
    except Exception as e:
        logging.error(f"Forward pass error: {e}")
    
    # Test backward pass
    try:
        logging.info("Testing backward pass...")
        
        # Create grad_output
        grad_output = torch.randn(M, N * G, dtype=torch.bfloat16, device=device)
        logging.info(f"Gradient shape: {grad_output.shape}")
        
        # Check autotuner configs
        from tgrouped_gemm_backwards import _CONFIGS, early_config_prune
        logging.info(f"Number of autotuner configs: {len(_CONFIGS)}")
        
        # Debug early_config_prune
        test_name_args = {
            "G": G,
            "M_bucket": M,
            "N": N,
            "K": K,
            "grad_x_ptr": x,  # Just for element_size()
        }
        
        pruned_configs = early_config_prune(_CONFIGS, test_name_args)
        logging.info(f"Number of pruned configs: {len(pruned_configs)}")
        
        if len(pruned_configs) == 0:
            logging.error("All configs were pruned! This will cause min() error")
            
            # Try to understand why configs are being pruned
            device = torch.cuda.current_device()
            from triton.runtime import driver
            max_shared_memory = driver.active.utils.get_device_properties(device)["max_shared_mem"]
            logging.info(f"Max shared memory: {max_shared_memory}")
            
            # Try with a minimal config
            from triton import Config
            minimal_config = Config(
                {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
                num_stages=1,
                num_warps=4,
                num_ctas=1
            )
            
            test_dtsize = x.element_size()
            req_shared_mem = (32 + 32) * 32 * 1 * test_dtsize
            logging.info(f"Required shared memory for minimal config: {req_shared_mem}")
            logging.info(f"Is minimal config valid? {req_shared_mem <= max_shared_memory}")
        
        # Now test the actual backward pass
        x_back = x.clone().detach().requires_grad_(True)
        w_back = w.clone().detach().requires_grad_(True)
        
        # Forward pass
        result = grouped_gemm(x_back, w_back, m_sizes)
        
        # Backward pass
        try:
            result.backward(grad_output)
            logging.info("Backward pass completed successfully")
            logging.info(f"grad_x shape: {x_back.grad.shape}")
            logging.info(f"grad_w shape: {w_back.grad.shape}")
        except Exception as e:
            logging.error(f"Error during .backward(): {e}")
            
            # Try direct call to the backward function
            try:
                grad_x, grad_w = grouped_gemm_backward(grad_output, x_back, w_back, m_sizes)
                logging.info("Direct backward call completed successfully")
                logging.info(f"grad_x shape: {grad_x.shape}")
                logging.info(f"grad_w shape: {grad_w.shape}")
            except Exception as e:
                logging.error(f"Error during direct backward call: {e}")
    
    except Exception as e:
        logging.error(f"Backward pass setup error: {e}")

if __name__ == "__main__":
    debug_grouped_gemm()
