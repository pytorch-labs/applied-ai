import argparse
import time
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM 

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def benchmark_generation_speed(model, tokenizer, prompt, batch_size, device, num_passes=5):

    token_dict = tokenizer([prompt] * batch_size, return_tensors="pt", padding="longest").to(device)

    total_generation_time = 0
    total_num_generated_tokens = 0

    # Warmup
    logger.info("Starting warmup...")
    for _ in tqdm(range(4), desc="Warmup", leave=False):
        with torch.inference_mode():
            _ = model.generate(**token_dict, min_length=30, max_length=30)

    logger.info("Starting benchmark...")
    with tqdm(range(num_passes), desc="Benchmark Passes") as pbar:
        for pass_num in pbar:
            token_dict = tokenizer([prompt] * batch_size, return_tensors="pt", padding="longest").to(device)

            start = time.time()
            with torch.inference_mode():
                outputs_ids = model.generate(**token_dict, min_length=30, max_length=30)
            end = time.time()

            generation_time = end - start
            num_generated_tokens = sum(len(output_ids) for output_ids in outputs_ids) - batch_size * len(token_dict['input_ids'][0])
            tokens_per_second = num_generated_tokens / generation_time

            total_generation_time += generation_time
            total_num_generated_tokens += num_generated_tokens

            # Update tqdm post-fix with current iteration results
            pbar.set_postfix({"Time (s)": f"{generation_time:.2f}", "Tokens/s": f"{tokens_per_second:.2f}"})

    # Calculate average statistics
    avg_generation_time = total_generation_time / num_passes
    avg_tokens_per_second = total_num_generated_tokens / total_generation_time
    avg_num_generated_tokens = total_num_generated_tokens / num_passes

    # Log average statistics
    logger.info(f"Batch size: {batch_size}, Avg Time: {avg_generation_time:.2f}s, Avg Tokens/s: {avg_tokens_per_second:.2f}, Avg Total tokens: {avg_num_generated_tokens}")
    return avg_generation_time, avg_tokens_per_second, avg_num_generated_tokens



def main():
    parser = argparse.ArgumentParser(description='Benchmark Llama-70B')
    parser.add_argument('--use_triton', type=lambda x: (str(x).lower() == 'true'), help='use Triton Kernel')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for the benchmark')
    args = parser.parse_args()

    device = "cuda:5"  
    quantized_model_dir = '/net/storage149/autofs/css22/ccyang/fm-models/llama-gptq/gptq_output_act0_grp128_bluewiki' 
    
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"

    if args.use_triton:
        torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=device, inject_fused_attention=False, inject_fused_mlp=False,
                                                use_triton=args.use_triton, disable_exllamaV2=True, low_cpu_mem_usage=True, warmup_triton=False)
    else:
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=device, inject_fused_attention=False, inject_fused_mlp=False,
                                            use_triton=False, disable_exllamaV2=False, low_cpu_mem_usage=True, warmup_triton=False)
        
    model = torch.compile(model, mode="reduce-overhead")
    benchmark_generation_speed(model, tokenizer, "auto-gptq is a", args.batch_size, device)

if __name__ == "__main__":
    main()
