#include <iostream>
#include <ATen/ATen.h>
#include <bitset>
#include <cstdlib>
#include <time.h>

int float_experiment() {
  // investigation of the floating point format
  float a_float = 0.999;
  std::cout << "float number: " << a_float << "\n";
  unsigned int a_int = *reinterpret_cast<unsigned int*>(&a_float);
  std::bitset<32> a_bitstring(a_int);
  unsigned int sign_bit = a_int >> 31;
  std::cout << "sign bit: " << sign_bit << "\n";
  unsigned int exp = a_int << 1 >> 24;
  std::bitset<8> exp_bits(exp);
  std::cout << "exponent: " << ((int)exp)-126 << " exponent bits: " << exp_bits << "\n";
  unsigned int man = a_int << 9 >> 9;
  std::bitset<32> man_bits(man);
  std::cout << "mantissa bits: " << man_bits << "\n";
  // stochastic rounding
  int wl = 3; // word length
  wl = wl - 1; // sign bit and virtual bit
  srand(time(NULL));
  unsigned int r = ((unsigned int)rand()) << (9+wl) >> (9+wl);
  // unsigned int r = ((unsigned int) -1) >> 31 << 31 >> (9+wl);
  std::bitset<32> random_bits(r);
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

int clip_exponent_experiment() {
  int wl = 2;
  unsigned int max_man = (unsigned int) -1 << (32-wl) >> 9;
  std::bitset<32> max_man_bits(max_man);
  std::cout << "max man bits: " << max_man_bits << "\n";
}

int max_min_experiment(int wl, int fl) {
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-ldexp(1.0, sigma);
  std::cout << "tmax: " << t_max << " tmin: " << t_min << "\n";
}

int block_float_experiment(float max_float, float to_quantize_float) {
  unsigned int max_num = *reinterpret_cast<unsigned int*>(&max_float);
  unsigned int to_quantize = *reinterpret_cast<unsigned int*>(&to_quantize_float);
  unsigned int max_exp = max_num << 1 >> 24;
  std::bitset<32> max_exp_bits(max_exp);
  // std::cout << "max exponent       :             " << max_exp_bits << "\n";
  unsigned int to_exp = to_quantize << 1 >> 24;
  std::bitset<32> to_exp_bits(to_exp);
  // std::cout << "to exponent        :             " << to_exp_bits << "\n";
  int wl = 4;
  int man = wl-2; //counting a virtual bit
  int offset = max_exp-to_exp;
  std::cout << "man:" << man << "\n";
  std::cout << "max_exp:" << (int)max_exp-127 << "\n";
  std::cout << "to_exp:" << (int)to_exp-127 << "\n";
  std::cout << "offset:" << offset << "\n";
  if ((man-offset) < 0) {
    float quantized_float = 0;
    std::cout << "quantized float:             " << quantized_float << "\n";
  } else {
    srand(time(NULL));
    unsigned int man_mask = ((1 << (23-(man-offset)))- 1);
    // unsigned int r = ((unsigned int)rand()) & man_mask;
    unsigned int r = 1 << (23-(man-offset)-1);
    // std::bitset<32> max_bits(max_num);
    std::bitset<24> to_bits(to_quantize);
    std::bitset<24> rand_bits(r);
    // std::cout << "number to quantize :             " << to_bits << "\n";
    // std::cout << "masked random      :             " << rand_bits << "\n";
    unsigned int added = (to_quantize+r);
    unsigned int new_exp = added << 1 >> 24;
    offset = max_exp-new_exp;
    std::bitset<32> added_bits(added);
    // std::cout << "after r added      :             " << added_bits << "\n";
    // std::cout << "offset:" << offset << "\n";
    if (offset > 0) {
      unsigned int added_man = added << (9) >> 1;
      unsigned int real_man = ((1 << 31) | added_man) & ~((unsigned int) -1 << (wl-1-offset) >> (wl-1-offset));
      unsigned int block_man = real_man >> (8+offset); // off by 1 because we add the virtual bit
      std::bitset<23> added_man_bits(real_man);
      // std::cout << "after r added man  :             " << added_man_bits << "\n";
      unsigned int sign = to_quantize >> 31 << 31;
      unsigned int quantized = (sign | max_exp << 23) | block_man;
      unsigned int virtual_offset = sign | max_exp << 23;
      std::bitset<32> quant_bits(quantized);
      std::cout << "quantized num      :             " << quant_bits << "\n";
      float quantized_float = *reinterpret_cast<float*>(&quantized);
      float virtual_offset_float = *reinterpret_cast<float*>(&virtual_offset);
      std::cout << "quantized float:             " << quantized_float-virtual_offset_float << "\n";
    } else {
      unsigned int quantized = added & ~man_mask;
      float quantized_float = *reinterpret_cast<float*>(&quantized);
      std::bitset<32> quant_bits(quantized);
      std::cout << "quantized num      :             " << quant_bits << "\n";
      std::cout << "quantized float:             " << quantized_float << "\n";
    }
  }
}

int main() {
  // block_float_experiment(0.25, 0.87);
  // block_float_experiment(0.5, 0.87);
  // block_float_experiment(1, 0.87);
  // block_float_experiment(2, 0.87);
  block_float_experiment(4, 0.87);
  // block_float_experiment(8, 0.87);
}
