#include <iostream>
#include <ATen/ATen.h>
#include <bitset>
#include <cstdlib>
#include <time.h>

int float_experiment() {
  // investigation of the floating point format
  float a_float = -0.19;
  std::cout << "float number: " << a_float << "\n";
  unsigned int a_int = *reinterpret_cast<unsigned int*>(&a_float);
  std::bitset<32> a_bitstring(a_int);
  unsigned int sign_bit = a_int >> 31;
  std::cout << "sign bit: " << sign_bit << "\n";
  unsigned int exp = a_int << 1 >> 24;
  std::bitset<8> exp_bits(exp);
  std::cout << "exponent: " << ((int)exp)-126 << " exponent bits: " << exp_bits << "\n";
  unsigned int man = a_int << 9 >> 9;
  std::bitset<23> man_bits(man);
  std::cout << "mantissa bits: " << man_bits << "\n";
  // stochastic rounding
  int wl = 2; // word length
  wl = wl - 2; // sign bit and virtual bit
  srand(time(NULL));
  unsigned int r = ((unsigned int)rand()) << (9+wl) >> (9+wl);
  std::bitset<23> random_bits(r);
  std::cout << "random bits:   " << random_bits << "\n";
  std::cout << "bit string before add: " << a_bitstring << "\n";
  unsigned int addr = a_int+r;
  std::bitset<32> addr_bits(addr);
  std::cout << "bit string after add:  " << addr_bits << "\n";
  unsigned int mask = ((unsigned int) -1) << (32-9-wl);
  std::bitset<32> mask_bits(mask);
  std::cout << "mask:                  " << mask_bits << "\n";
  unsigned int quantized = addr & mask;
  std::bitset<32> quantized_bits(quantized);
  std::cout << "quantized number:      " << quantized_bits << "\n";
  float quantized_float = *reinterpret_cast<float*>(&quantized);
  std::cout << "quantized float: " << quantized_float << "\n";
  // clip exponent
  unsigned int qexp = quantized << 1 >> 24;
  std::bitset<8> qexp_bits(qexp);
  std::cout << "exponent after qauntization: " << ((int)qexp-126) << "\n";
  std::cout << "exponent bits after qauntization: " << qexp_bits << "\n";
  if (qexp > exp) {
    int offset = 32-9-wl;
    unsigned int max_mask = (unsigned int) -1 >> offset << offset;
    std::bitset<32> max_mask_bits(max_mask);
    std::cout << "max mask: " << max_mask_bits << "\n";
    quantized = (a_int >> (32-9-wl) << (32-9-wl)) & max_mask;
    float clipped_quantized_float = *reinterpret_cast<float*>(&quantized);
    std::bitset<32> clip_quant_bits(quantized);
    std::cout << "quantized bits after clipping: " << clip_quant_bits << "\n";
    std::cout << "quantized number after clipping: " << clipped_quantized_float << "\n";
  }
  return 0;
}

// syntax experiment on how to generate random integers with ATen
int random_experiment() {
  std::cout << "create a random integer tensor" << "\n";
  int man_bits = 2;
  man_bits = man_bits-1;
  int offset = 1 << (32-9-man_bits);
  at::Tensor rand_ints = at::randint(offset, {4, 4}, at::device(at::kCPU).dtype(at::kInt));
  std::cout << rand_ints << "\n";
}

// syntax experiment on how to get the max number with regards to
// [exp_bits] and [man_bits]
int max_number_experiment() {
  int exp_bits = 8;
  unsigned int max_exp = (unsigned int)-1 << (32-exp_bits) >> (32-exp_bits);
  std::bitset<32> max_exp_bits(max_exp);
  std::cout << "max exponent bits: " << max_exp_bits << "\n";
  max_exp = max_exp << 23;
  std::bitset<32> max_exp_after_shift_bits(max_exp);
  std::cout << "max exponent bits after shift: " << max_exp_after_shift_bits << "\n";
  int man_bits = 3;
  man_bits = man_bits-1; // 1 virtual bit
  int offset = 32-9-man_bits;
  unsigned int max_man = (unsigned int) -1 << 9 >> 9 >> offset << offset;
  std::bitset<32> max_man_bits(max_man);
  std::cout << "max mantissa bits :            " << max_man_bits << "\n";
  unsigned int max_num = max_exp | max_man;
  unsigned int old_sign = (unsigned int) -1 >> 31 << 31; // simply assuming old number is negative
  std::bitset<32> old_sign_bits(old_sign);
  std::cout << "sign bit only is :             " << old_sign_bits << "\n";
}

int main() {
  max_number_experiment();
}
