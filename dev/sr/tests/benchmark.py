import torch
import stochastic_rounding_cuda
import numpy as np
import time
from tabulate import tabulate
import argparse

def measure_performance(func, input_tensor, warmup=10, repeats=100):
    """Measure performance of a function with proper CUDA synchronization"""
    # Warmup
    for _ in range(warmup):
        output = func(input_tensor)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(repeats):
        output = func(input_tensor)

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / repeats
    elements_per_second = input_tensor.numel() / avg_time
    return avg_time, elements_per_second

def benchmark_sizes(sizes=[1000, 10000, 100000, 1000000, 10000000]):
    """Benchmark different input sizes"""
    results = []

    for size in sizes:
        # Create input tensor
        x = torch.randn(size, device='cuda')

        # Measure stochastic rounding
        time_stoch, throughput_stoch = measure_performance(
            stochastic_rounding_cuda.stochastic_round_bf16, x)

        # Measure regular BF16 casting
        time_regular, throughput_regular = measure_performance(
            lambda t: t.to(torch.bfloat16), x)

        results.append([
            size,
            time_stoch * 1000,  # convert to ms
            throughput_stoch / 1e9,  # convert to GElements/s
            time_regular * 1000,
            throughput_regular / 1e9,
            throughput_regular / throughput_stoch  # speedup
        ])

    print("\nSize Comparison:")
    print(tabulate(results,
                  headers=['Size', 'Stoch Time (ms)', 'Stoch GE/s',
                          'Regular Time (ms)', 'Regular GE/s', 'Casting faster by'],
                  floatfmt='.3f'))

def benchmark_shapes(total_size=1000000):
    """Benchmark different tensor shapes with same total size"""
    shapes = [
        (total_size,),           # 1D
        (1000, total_size//1000),  # 2D
        (100, 100, total_size//10000),  # 3D
    ]

    results = []
    for shape in shapes:
        x = torch.randn(*shape, device='cuda')
        time_stoch, throughput_stoch = measure_performance(
            stochastic_rounding_cuda.stochastic_round_bf16, x)

        results.append([
            'x'.join(str(d) for d in shape),
            time_stoch * 1000,
            throughput_stoch / 1e9
        ])

    print("\nShape Comparison (same total size):")
    print(tabulate(results,
                  headers=['Shape', 'Time (ms)', 'GElements/s'],
                  floatfmt='.3f'))

def stress_test(duration=10):
    """Run a stress test for specified duration"""
    print(f"\nRunning stress test for {duration} seconds...")

    size = 1000000
    x = torch.randn(size, device='cuda')
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration:
        stochastic_rounding_cuda.stochastic_round_bf16(x)
        iterations += 1

    print(f"Completed {iterations} iterations without errors")
    print(f"Average throughput: {(iterations * size) / (duration * 1e9):.2f} GElements/s")

def memory_test(max_size=1e9):
    """Test memory scaling"""
    sizes = np.logspace(3, min(9, np.log10(max_size)), num=7, dtype=int)
    results = []

    for size in sizes:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(size, device='cuda')
            torch.cuda.synchronize()

            # Measure peak memory during operation
            torch.cuda.reset_peak_memory_stats()
            _ = stochastic_rounding_cuda.stochastic_round_bf16(x)
            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
            results.append([size, peak_memory])

        except RuntimeError as e:
            print(f"Out of memory at size {size}")
            break

    print("\nMemory Usage:")
    print(tabulate(results,
                  headers=['Size', 'Peak Memory (MB)'],
                  floatfmt='.2f'))

def main():
    parser = argparse.ArgumentParser(description='Benchmark stochastic rounding')
    parser.add_argument('--sizes', action='store_true', help='Run size benchmarks')
    parser.add_argument('--shapes', action='store_true', help='Run shape benchmarks')
    parser.add_argument('--stress', action='store_true', help='Run stress test')
    parser.add_argument('--memory', action='store_true', help='Run memory test')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')

    args = parser.parse_args()

    # Print device information
    device = torch.cuda.get_device_properties(0)
    print(f"\nRunning on: {device.name}")
    print(f"Compute Capability: {device.major}.{device.minor}")


    if args.all or args.sizes:
        benchmark_sizes()

    if args.all or args.shapes:
        benchmark_shapes()

    if args.all or args.stress:
        stress_test()

    if args.all or args.memory:
        memory_test()

if __name__ == '__main__':
    main()
