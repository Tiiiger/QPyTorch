#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))

__device__ __forceinline__ unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (temp << 1 >> 24); // single preciision, 1 sign bit, 23 mantissa bits
  return temp-127+1; // exponent offset and virtual bit
}

__device__ __forceinline__ unsigned int round_bitwise_stochastic(unsigned int target,
                                                                 unsigned int rand_prob,
                                                                 int man_bits) {
    unsigned int mask = (1 << (23-man_bits)) - 1;
    unsigned int add_r = target+(rand_prob & mask);
    unsigned int quantized = add_r & ~mask;
    return quantized;
}

__device__ __forceinline__ unsigned int round_bitwise_nearest(unsigned int target,
                                                              int man_bits) {
    unsigned int mask = (1 << (23-man_bits)) - 1;
    unsigned int rand_prob = 1 << (23-man_bits-1);
    unsigned int add_r = target+rand_prob;
    unsigned int quantized = add_r & ~mask;
    return quantized;
}

__device__ __forceinline__ unsigned int clip_exponent(int exp_bits, int man_bits,
                                                      unsigned int old_num,
                                                      unsigned int quantized_num) {
  // int offset = 32-9-man_bits; // float length minus sign bit and exponent bit add 1 virtual bit
  // unsigned int quantized_exponent_store = quantized_num << 1 >> 1 >> 23; // 1 sign bit, 23 mantissa bits
  // int quantized_exponent_real = (int) quantized_exponent_store - 126;
  // unsigned int max_exponent = (unsigned int) 1 << exp_bits;
  // if (quantized_exponent > max_exponent) {
  //   unsigned int max_man = (unsigned int ) -1 << (32-wl) >> 9; // 1 sign bit, 8 exponent bits
  //   unsigned int max_num = (max_exponent << 23) | max_man;
  //   unsigned int old_sign = old_num >> 31 << 31;
  //   quantized_num = old_sign | max_num;
  // }
  return quantized_num;
}