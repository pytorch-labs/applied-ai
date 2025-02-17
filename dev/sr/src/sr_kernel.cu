#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

// convert float to stochastically rounded __nv_bfloat16 (bf16)
__device__ __forceinline__ __nv_bfloat16 float_to_SR_bf16(float x,
                                                          curandState *state) {
  if (isnan(x) || isinf(x)) {
    return __float2bfloat16(x); // TODO - how to better handle?
  }

  // Decompose float to components
  union {
    float f;
    uint_32_t i;

  } u;

  u.f = x;

  // extract components
  uint32_t sign = u.i & 0x80000000;
  uint32_t exponent = u.i & 0x7F800000;
  uint32_t mantissa = u.i & 0x007FFFFF;

  // skip if zero
  if (exponent == 0 && mantissa == 0) {
    return __float2bfloat16(x);
  }

  // Determine rounding probability
  //  We are using the upper 8 bits to improve speed
  //  so rounding probs are 256 granularity.
  uint32_t upper_dropped_bits = (mantissa & 0x0000FF00) >> 8;
  float prob = float(upper_dropped_bits) / float(0x100);

  // random number drawing
  float rand = curand_uniform(state);

  // round up or not
  uint32_t rounded_bit = (rand < prob) ? 0x80 : 0;

  // Combine together
  uint32_t bf16_bits =
      sign | exponent | ((mantissa + (rounded_bit << 8)) >> 16);
  u.i = bf16_bits;
  return __float2bfloat16(u.f);
}
