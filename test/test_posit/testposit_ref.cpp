/*
 * g++ testpositnew.cpp -o testpositnew -std=c++11
 */
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t
typedef FP16_TYPE fp16;

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

#define _G_NBITS 4
#define _G_ESIZE 1

#define SIGN_MASK 0x8000
#define FLOAT_SIGN_MASK 0x80000000
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF
#define SECOND_BIT_MASK 0x4000
#define POSIT_INF 0x0000
#define POSIT_LIMB_ALL_BITS_SET 0xffff
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_SIZE 32
#define FLOAT_EXPONENT_MASK 0x7f800000
#define FLOAT_FRACTION_MASK 0x007fffff
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8
#define TEMP_TYPE uint64_t
#define UNSIGNED_LONG_LONG_SIZE 64
#define EDP_ACC_SIZE 63
#define POSIT_EXP_SHIFT 41 //64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9
#define POSIT_LENGTH_PLUS_ONE 17


#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define _G_INFP 32768

#define _G_POSIT_SHIFT_AMOUNT 12
#define _G_MAXREALP 28672
#define _G_MINREALP 4096
#define POSIT_EXTRA_BITS_SHIFT 61 // 64 - _G_NBITS + 1
#define POSIT_EXTRA_BITS_MASK 0x0FFFFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x1000000000000000
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 16.0
#define _G_MINREAL 0.0625
#define _G_MAXREAL_INT 0x41800000
#define _G_MINREAL_INT 0x3d800000

float fp16tofp32(fp16 p) {
  union Bits v;

  // get sign
  bool sign = p & SIGN_MASK;
  p = (p ^ -sign) + sign;

  // get the regime sign
  bool regime_sign = p & SECOND_BIT_MASK;

  // get regime
  v.ui = p << POSIT_LENGTH_PLUS_ONE;
  //int regime_length = (__clz(v.ui) & -!regime_sign) + (__clz(~v.ui) & -regime_sign);
  int regime_length;
  if(regime_sign)
    regime_length = (__builtin_clz(~v.ui));
  else
    regime_length = (__builtin_clz(v.ui));
  int regime = (regime_length - regime_sign) << _G_ESIZE;
  regime = (regime ^ -regime_sign) + regime_sign;

  // assemble
  v.ui <<= (regime_length + 1);
  v.ui >>= (FLOAT_SIGN_PLUS_EXP_LENGTH - _G_ESIZE);
  v.ui += ((SINGLE_PRECISION_BIAS - regime) << FLOAT_EXPONENT_SHIFT);

  v.si ^= (FLOAT_INF ^ v.si) & -(p == _G_INFP);
  v.si ^= (0 ^ v.si) & -(p == 0);

  v.ui |= (sign << FLOAT_SIGN_SHIFT);
  return v.f;

}

fp16 fp32tofp16(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;

  p ^= (p ^_G_MAXREALP) & -(v.si >= _G_MAXREAL_INT);
  p ^= (p ^ _G_INFP) & -(v.si >= FLOAT_INF);
  p ^= (p ^ _G_MINREALP) & -(v.si != 0 && v.si <= _G_MINREAL_INT);

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get exponent sign
  bool exp_sign = !(v.ui >> FLOAT_EXP_SIGN_SHIFT);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_ESIZE + 1)) | (exp & POSIT_EXPONENT_MASK);;
  //if exponent is negative
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign) >> ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (POSIT_EXP_SHIFT - regime_and_exp_length));
  fp16 temp_p = (regime_and_exp >> POSIT_EXTRA_BITS_SHIFT);

  //round
  temp_p += (bool) (regime_and_exp & POSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & POSIT_EXTRA_BITS_MASK));
 if (_G_NBITS != 16)
    temp_p <<= _G_POSIT_SHIFT_AMOUNT;

  p ^= (temp_p ^ p) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  p = (p ^ -sign) + sign;

  return p;
}

int main() {
	fp16 p = fp32tofp16(-15.0);
    printf("%d\n", p);
	printf("%f\n", fp16tofp32(p));
}
