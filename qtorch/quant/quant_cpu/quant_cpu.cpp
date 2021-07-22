#include <torch/torch.h>
#include <assert.h>
#include <random>
#include <tuple>
#include "quant_cpu.h"

using namespace at;

enum Mode
{
  rNearest,
  rStochastic
};

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x);

#define RFLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int *>(x))
#define RBITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))
#define FLOAT_TO_BITS(f, i)     \
  assert(sizeof f == sizeof i); \
  std::memcpy(&i, &f, sizeof i)
#define BITS_TO_FLOAT(i, f)     \
  assert(sizeof f == sizeof i); \
  std::memcpy(&f, &i, sizeof f)

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0);

template <typename T>
T clamp_helper(T a, T min, T max)
{
  if (a > max)
    return max;
  else if (a < min)
    return min;
  else
    return a;
}

template <typename T>
T clamp_mask_helper(T a, T min, T max, uint8_t *mask)
{
  if (a > max)
  {
    *mask = 1;
    return max;
  }
  else if (a < min)
  {
    *mask = 1;
    return min;
  }
  else
    return a;
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl, bool symmetric)
{
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data_ptr<float>();
  auto r_array = r.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  auto m = zeros_like(a, torch::CPU(kByte));
  auto m_array = m.data_ptr<uint8_t>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    o_array[i] = clamp_mask_helper<float>(o_array[i], t_min, t_max, m_array + i);
  }
  return std::make_tuple(o, m);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl, bool symmetric)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  auto m = zeros_like(a, torch::CPU(kByte));
  auto m_array = m.data_ptr<uint8_t>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], 0.5, sigma);
    o_array[i] = clamp_mask_helper<float>(o_array[i], t_min, t_max, m_array + i);
  }
  return std::make_tuple(o, m);
}

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool clamp, bool symmetric)
{
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data_ptr<float>();
  auto r_array = r.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    if (clamp)
    {
      o_array[i] = clamp_helper(o_array[i], t_min, t_max);
    }
  }
  return o;
}

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool clamp, bool symmetric)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], 0.5, sigma);
    if (clamp)
    {
      o_array[i] = clamp_helper(o_array[i], t_min, t_max);
    }
  }
  return o;
}

unsigned int round_bitwise(unsigned int target, int man_bits, Mode rounding)
{
  unsigned int mask = (1 << (23 - man_bits)) - 1;
  unsigned int rand_prob;
  if (rounding == rStochastic)
  {
    rand_prob = (dis(gen)) & mask;
  }
  else
  {
    rand_prob = 1 << (23 - man_bits - 1);
  }
  unsigned int add_r = target + rand_prob;
  unsigned int quantized = add_r & ~mask;
  return quantized;
}

void block_quantize_helper(float *input, float *output, float *max_elem,
                           int wl, int size, Mode rounding)
{
  for (int64_t i = 0; i < size; i++)
  {

    unsigned int max_num;
    FLOAT_TO_BITS(max_elem[i], max_num);
    unsigned int max_exp = max_num << 1 >> 24 << 23;
    float base_float;
    BITS_TO_FLOAT(max_exp, base_float);
    base_float *= 6;

    float target_rebase = input[i] + base_float;
    unsigned int target_bits;
    FLOAT_TO_BITS(target_rebase, target_bits);
    unsigned int quantized_bits = round_bitwise(target_bits, wl, rounding); // -1 sign, -1 virtual, +2 base
    float quantized_rebase;
    BITS_TO_FLOAT(quantized_bits, quantized_rebase);
    float quantized = quantized_rebase - base_float;

    unsigned int quantize_bits;
    FLOAT_TO_BITS(quantized, quantize_bits);
    unsigned int clip_quantize = clip_max_exponent(wl - 2, max_exp, quantize_bits);
    BITS_TO_FLOAT(clip_quantize, quantized);

    output[i] = quantized;
  }
}

Tensor get_max_entry(Tensor a, int dim)
{
  Tensor max_entry;
  if (dim == -1)
  {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  }
  else if (dim == 0)
  {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.max(1, true)).abs().expand_as(input_view).view_as(a).contiguous();
  }
  else
  {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view = input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.max(1, true)).abs().expand_as(input_view).view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = get_max_entry(a, dim);
  auto max_elem = max_entry.data_ptr<float>();
  block_quantize_helper(a_array, o_array, max_elem, wl, size, rNearest);
  return o;
}

Tensor block_quantize_stochastic(Tensor a, int wl, int dim)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = get_max_entry(a, dim);
  auto max_elem = max_entry.data_ptr<float>();
  // std::srand(time(0));
  block_quantize_helper(a_array, o_array, max_elem, wl, size, rStochastic);
  return o;
}

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits)
{
  // use external random number right now
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  for (int64_t i = 0; i < size; i++)
  {
    unsigned int target;
    FLOAT_TO_BITS(a_array[i], target);
    unsigned int quantize_bits = round_bitwise(target, man_bits, rStochastic);
    quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
    float quantized;
    BITS_TO_FLOAT(quantize_bits, quantized);
    o_array[i] = quantized;
  }
  return o;
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  for (int64_t i = 0; i < size; i++)
  {
    unsigned int target;
    FLOAT_TO_BITS(a_array[i], target);
    unsigned int quantize_bits = round_bitwise(target, man_bits, rNearest);
    quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
    float quantized;
    BITS_TO_FLOAT(quantize_bits, quantized);
    o_array[i] = quantized;
  }
  return o;
}

#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t



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

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

typedef FP16_TYPE fp16;
#define _G_POSIT_SHIFT_AMOUNT   int32_constants[0]
#define _G_MAXREALP             int32_constants[1]
#define _G_MINREALP             int32_constants[2]
#define POSIT_EXTRA_BITS_SHIFT  int32_constants[3]
#define _G_USEED                int32_constants[4]
#define _G_USEED_ZEROS          int32_constants[5]
#define POSIT_EXPONENT_MASK     int32_constants[6]
#define _G_MAXREAL_INT          int32_constants[7]
#define _G_MINREAL_INT          int32_constants[8]
#define _G_NBITS                int32_constants[9]
#define _G_ESIZE                int32_constants[10]

#define POSIT_EXTRA_BITS_MASK   int64_constants[0]
#define POSIT_HALFWAY_BIT_MASK  int64_constants[1]

void generate_posit_constants(int nsize, int es, uint32_t* int32_constants, uint64_t* int64_constants) {
  //local vars have the same name as global constant vars, confusing but less likely error can happen here.
  //ugly but it's the traightforward conversion from the original #define macroes;
  //todo: make this one less messy
  _G_NBITS = nsize;
  _G_ESIZE = es;
  if (nsize <= 16 ) {
    _G_POSIT_SHIFT_AMOUNT = FP16_LIMB_SIZE - nsize;
    _G_MAXREALP = ((1 << (nsize - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT;
    _G_MINREALP = 1 << _G_POSIT_SHIFT_AMOUNT;
    POSIT_EXTRA_BITS_SHIFT = UNSIGNED_LONG_LONG_SIZE - nsize + 1;
    POSIT_EXTRA_BITS_MASK = (1UL << (UNSIGNED_LONG_LONG_SIZE - nsize)) - 1;
    POSIT_HALFWAY_BIT_MASK = 1UL << (UNSIGNED_LONG_LONG_SIZE - nsize);
    _G_USEED = 1 << (1 << es);
    _G_USEED_ZEROS = (1 << es);
    POSIT_EXPONENT_MASK = _G_USEED_ZEROS - 1;
    _G_MAXREAL_INT = ((_G_USEED_ZEROS * (nsize - 2)) + SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
    _G_MINREAL_INT = ((_G_USEED_ZEROS * (2 - nsize)) + SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
  } else {
    printf("unexpected posit config\n");
    exit(1);
  }
};

float fp16tofp32(fp16 p, uint32_t* int32_constants, uint64_t* int64_constants) {
	union Bits v;

	// get sign
	bool sign = p & SIGN_MASK;
	p = (p ^ -sign) + sign;

	// get the regime sign
	bool regime_sign = p & SECOND_BIT_MASK;

	// get regime
	v.ui = p << POSIT_LENGTH_PLUS_ONE;
	//int regime_length = (__builtin_clz(v.ui) & -!regime_sign) + (__builtin_clz(~v.ui) & -regime_sign);
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

fp16 fp32tofp16(float f,  uint32_t* int32_constants, uint64_t* int64_constants) {
	fp16 p = 0;
	union Bits v;
	v.f = f;
	bool sign = v.ui & FLOAT_SIGN_MASK;
	v.ui &= 0x7FFFFFFF;

#ifdef FLOAT_ROUNDING
	uint16_t roundSign = sign << 15;
	if(v.ui > _G_MAXREAL_INT)
		return _G_INFP | roundSign;
	if(v.ui < _G_MINREAL_INT)
		return 0;
#endif
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

Tensor posit_quantize_nearest(Tensor a, int nsize, int es, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();
  uint32_t	int32_constants[ 11 ];
  uint64_t	int64_constants[ 2 ];

  generate_posit_constants(nsize, es, int32_constants, int64_constants);


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i]*scale;

    fp16 temp = fp32tofp16(temp_input, int32_constants, int64_constants);
    temp_input = fp16tofp32(temp, int32_constants, int64_constants);

    o_array[i] = temp_input/scale;

  }

  return o;
}

fp16 compute_sigmoid(fp16 p) {
    p ^= 0x8000;
    return p >> 2;
}

Tensor posit_sigmoid(Tensor a, int nsize, int es, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();
  uint32_t	int32_constants[ 11 ];
  uint64_t	int64_constants[ 2 ];
  //only works on nsize = 8 or 16
  generate_posit_constants(nsize, 0, int32_constants, int64_constants);


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i];//*scale;

    fp16 temp = fp32tofp16(temp_input, int32_constants, int64_constants);

    temp = compute_sigmoid (temp);

    temp_input = fp16tofp32(temp, int32_constants, int64_constants);

    o_array[i] = temp_input;///scale;

  }

  return o;
}


Tensor posit_tanh(Tensor a, int nsize, int es, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();
  uint32_t	int32_constants[ 11 ];
  uint64_t	int64_constants[ 2 ];
  //only works on nsize = 8 or 16
  generate_posit_constants(nsize, 0, int32_constants, int64_constants);


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i];//*scale;
    //tanh(x)=2g(2x)−1
    fp16 temp = fp32tofp16(2*temp_input, int32_constants, int64_constants);

    temp = compute_sigmoid (temp);

    temp_input = fp16tofp32(temp, int32_constants, int64_constants);

    temp_input = temp_input * 2 - 1 ;

    o_array[i] = temp_input;///scale;

  }

  return o;
}


/* // Deprecated, use new enhanced version with only add/substract below
Tensor posit_tanh_enhanced(Tensor a, int nsize, int es, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();
  uint32_t	int32_constants[ 11 ];
  uint64_t	int64_constants[ 2 ];
  //only works on nsize = 8 or 16
  generate_posit_constants(nsize, 0, int32_constants, int64_constants);


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i];//*scale;
    //tanh(x)=2g(2x)−1
    fp16 temp = fp32tofp16(2*temp_input, int32_constants, int64_constants);

    temp = compute_sigmoid (temp);

    temp_input = fp16tofp32(temp, int32_constants, int64_constants);

    temp_input = temp_input * 2 - 1 ;

      if (temp_input > 0.6)
          temp_input = temp_input*1.07;

      if (temp_input < -0.6)
          temp_input = temp_input*1.07;

      if (temp_input > 1)
          temp_input = 1;
      if (temp_input < -1)
          temp_input = -1;



      o_array[i] = temp_input;///scale;

  }

  return o;
}
*/

Tensor posit_tanh_enhanced(Tensor a, int nsize, int es, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();
  uint32_t	int32_constants[ 11 ];
  uint64_t	int64_constants[ 2 ];
  //only works on nsize = 8 or 16
  generate_posit_constants(nsize, 0, int32_constants, int64_constants);


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i];//*scale;
    //tanh(x)=2g(2x)−1
    fp16 temp = fp32tofp16(2*temp_input, int32_constants, int64_constants);

    temp = compute_sigmoid (temp);

    temp_input = fp16tofp32(temp, int32_constants, int64_constants);

    temp_input = temp_input * 2 - 1 ;

      if (temp_input > 0.7583)
          temp_input = temp_input+0.06795;

      if (temp_input < -0.7583)
          temp_input = temp_input-0.06795;

      if (temp_input > 1)
          temp_input = 1;
      if (temp_input < -1)
          temp_input = -1;



      o_array[i] = temp_input;///scale;

  }

  return o;
}


float new_format_quantize_nearest(float input){
    float constants[32] = {1.0/65536, 1.0/32768, 1.0/16384, 1.0/8192, 1.0/4096, 1.0/2048, 1.0/1024, 1.0/512, 1.0/256, 1.0/128,
               3.0/256, 1.0/64,  5.0/256 , 3.0/128,  7.0/256, 1.0/32, 9.0/256, 5.0/128, 3.0/64, 7.0/128,
               1.0/16,  9.0/128, 5.0/64, 3.0/32,    7.0/64,    1.0/8, 9.0/64, 3.0/16, 1.0/4, 3.0/8, 1.0/2, 1.0};
    float result = 0.0;
    if (input != 0.0){

      float min_abs_err = 1e5;
      float min_constant = 0.0;
      for (int i = 0; i<32; i ++){
          float abs_err = fabs(constants[i] - fabs(input));
          if(abs_err < min_abs_err){
             min_abs_err = abs_err;
             min_constant = constants[i];
          }

      }

      if (input < 0)
          result = - min_constant;
      else
          result = min_constant;
    }

    return result;

}

/*custom table lookup with given configurable constants codebook table having contant_size elements*/
float configurable_table_quantize_nearest(float input, float* constants, int constant_size){

    float result = 0.0;
    if (input != 0.0){

      float min_abs_err = 1e5;
      float min_constant = 0.0;
      for (int i = 0; i<constant_size; i ++){
          float abs_err = fabs(constants[i] - fabs(input));
          if(abs_err < min_abs_err){
             min_abs_err = abs_err;
             min_constant = constants[i];
          }

      }

      if (input < 0)
          result = - min_constant;
      else
          result = min_constant;
    }

    return result;

}

/*custom table lookup with given configurable constants codebook table having contant_size elements*/
float configurable_table_quantize_rounding_hint_f(float input, float* constants, float* rounding_hints, int constant_size){

    float result = 0.0;
    if (input != 0.0){
      float min_constant = 0.0;
      for (int i = 0; i<constant_size; i ++){
          //float abs_err = fabs(constants[i] - fabs(input));
          if (fabs(input) > rounding_hints[i])
            min_constant = constants[i];
      }

      if (input < 0)
          result = - min_constant;
      else
          result = min_constant;
    }

    return result;

}


float act_format_quantize_nearest(float input){

    float constants[32] = {1.0/4096, 1.0/2048, 1.0/1024, 1.0/512, 1.0/256, 1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 3.0/16,
                           1.0/4, 5.0/16, 3.0/8, 7.0/16, 1.0/2, 9.0/16, 5.0/8, 3.0/4, 7.0/8, 1.0, 9.0/8, 5.0/4, 3.0/2,
                           7.0/4, 2.0, 9.0/4, 3.0, 4.0, 6.0, 8.0, 16.0};
    float result = 0.0;
    if (input != 0.0){

      float min_abs_err = 1e5;
      float min_constant = 0.0;
      for (int i = 0; i<32; i ++){
          float abs_err = fabs(constants[i] - fabs(input));
          if(abs_err < min_abs_err){
             min_abs_err = abs_err;
             min_constant = constants[i];
          }

      }

      if (input < 0)
          result = - min_constant;
      else
          result = min_constant;
    }

    return result;

}

Tensor new_format_quantize(Tensor a, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i]*scale;

    temp_input = new_format_quantize_nearest(temp_input);

    o_array[i] = temp_input/scale;

  }

  return o;
}

Tensor act_format_quantize(Tensor a, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();


  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i]*scale;

    temp_input = act_format_quantize_nearest(temp_input);

    o_array[i] = temp_input/scale;

  }

  return o;
}

Tensor configurable_table_quantize(Tensor a, Tensor lookup_table, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  int table_size = lookup_table.numel();
  auto contants = lookup_table.data_ptr<float>();

  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i]*scale;

    temp_input = configurable_table_quantize_nearest(temp_input, contants, table_size);

    o_array[i] = temp_input/scale;

  }

  return o;
}

Tensor configurable_table_quantize_rounding_hint(Tensor a, Tensor lookup_table, Tensor rounding_hint, float scale)
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  int table_size = lookup_table.numel();

  auto contants = lookup_table.data_ptr<float>();

  auto rounding_hints = rounding_hint.data_ptr<float>();

  for (int64_t i = 0; i < size; i++)
  {
    float temp_input = a_array[i]*scale;

    temp_input = configurable_table_quantize_rounding_hint_f (temp_input, contants, rounding_hints, table_size);

    o_array[i] = temp_input/scale;

  }

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fixed_point_quantize_stochastic_mask", &fixed_point_quantize_stochastic_mask, "Fixed Point Number Stochastic Quantization with Mask (CPU)");
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CPU)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CPU)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask, "Fixed Point Number Nearest Quantization with Mask (CPU)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CPU)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def("posit_quantize_nearest", &posit_quantize_nearest, "Low-Bitwidth Posit Quantization (CPU)");
  m.def("posit_sigmoid", &posit_sigmoid, "Low-Bitwidth Posit Sigmoid (CPU)");
  m.def("posit_tanh", &posit_tanh, "Low-Bitwidth Posit Tanh (CPU)");
  m.def("posit_tanh_enhanced", &posit_tanh_enhanced, "Low-Bitwidth Posit Tanh (CPU)");
  m.def("new_format_quantize", &new_format_quantize, "New table-lookup Format (CPU)");
  m.def("act_format_quantize", &act_format_quantize, "New table-lookup Format (Activation CPU)");
  m.def("configurable_table_quantize", &configurable_table_quantize, "Configurable table-lookup Format (CPU)");
  m.def("configurable_table_quantize_rounding_hint", &configurable_table_quantize_rounding_hint, "Configurable table-lookup Format with hints for rounding for every interval (CPU)");
//  m.def("posit_tanh_enhanced2", &posit_tanh_enhanced2, "Low-Bitwidth Posit Tanh (CPU)");
}
