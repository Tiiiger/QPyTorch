#include <stdint.h>

unsigned int clip_exponent(int exp_bits, int man_bits, unsigned int old_num,
                           unsigned int quantized_num);

unsigned int clip_max_exponent(int man_bits,
                               unsigned int max_exponent,
                               unsigned int quantized_num);

template <typename T>
T clamp_helper(T a, T min, T max);

template <typename T>
T clamp_mask_helper(T a, T min, T max, uint8_t *mask);

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max);

float round(float a, float r, int sigma);
