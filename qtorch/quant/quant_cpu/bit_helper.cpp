#include "quant_cpu.h"
#include "stdio.h"

unsigned int clip_exponent(int exp_bits, int man_bits, unsigned int old_num,
                           unsigned int quantized_num) {
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store =
      quantized_num << 1 >> 1 >> 23; // 1 sign bit, 23 mantissa bits
  int max_exponent_store =
      (1 << (exp_bits - 1)) +
      127; // we are not reserving an exponent bit for infinity, nan, etc
  // Clippping Value Up
  if (quantized_exponent_store > max_exponent_store) {
    unsigned int max_man =
        (unsigned int)-1 << 9 >>
        9 >> (23 - man_bits)
                 << (23 -
                     man_bits); // 1 sign bit, 8 exponent bits, 1 virtual bit
    unsigned int max_num = ((unsigned int)max_exponent_store << 23) | max_man;
    unsigned int old_sign = old_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}

unsigned int clip_max_exponent(int man_bits, unsigned int max_exponent,
                               unsigned int quantized_num) {
  unsigned int quantized_exponent =
      quantized_num << 1 >> 24 << 23; // 1 sign bit, 23 mantissa bits
  if (quantized_exponent > max_exponent) {
    unsigned int max_man =
        (unsigned int)-1 << 9 >> 9 >>
        (23 - man_bits) << (23 - man_bits); // 1 sign bit, 8 exponent bits
    unsigned int max_num = max_exponent | max_man;
    unsigned int old_sign = quantized_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}
