/**
 * quantize a matrix of real numbers into fixed point numbers with
 * word length [wl] and fractional bits [fl].
 **/
void fixed_point_quantize(float *a, float *r, int size, int wl, int fl);

/**
 * quantize a matrix of real numbers into block floating point numbers with
 * word length [wl]. Treating each matrix as a block. 
 **/
void fixed_point_quantize(float *a, float *r, int size, int wl);


