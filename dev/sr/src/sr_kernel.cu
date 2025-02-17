#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Performance-optimized utility function
__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float x, curandState *state) {
  // Use intrinsics for special case checking
  if (__isnanf(x) || __isinff(x)) {
    return __float2bfloat16(x); // TODO - is this sufficient to just return...?
  }

  int x_int = __float_as_int(x);

  // Extract components
  uint32_t sign = x_int & 0x80000000;
  uint32_t exp = x_int & 0x7F800000;
  uint32_t mant = x_int & 0x007FFFFF;

  // Handle special case - zero
  if (exp == 0 && mant == 0) {
    return __float2bfloat16(x);
  }

  // Calculate rounding probability using upper 8 bits of the 16 dropped bits
  uint32_t upper_dropped_bits = (mant & 0x0000FF00) >> 8;

  // Add epsilon to avoid potential division by zero
  const float EPSILON = 1e-7f;
  float prob = float(upper_dropped_bits) / (float(0x100) + EPSILON);

  // Generate random number between 0 and 1
  float rand = curand_uniform(state);

  // Round up if random number is less than probability
  uint32_t round_bit = (rand < prob) ? 0x80 : 0;

  // Combine components with stochastic rounding
  uint32_t bf16_bits = sign | exp | ((mant + (round_bit << 8)) >> 16);

  // Convert back using intrinsic
  return __float2bfloat16(__int_as_float(bf16_bits));
}

// Calculate optimal block size based on GPU properties
__host__ int getOptimalBlockSize() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return min(prop.maxThreadsPerBlock,
             256); // TODO - this is a hardcoded estimate...
}
