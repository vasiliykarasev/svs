#ifndef BCV_BASIC_H_
#define BCV_BASIC_H_

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include "bcv_utils.h"

#ifdef HAVE_SSE
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#ifdef HAVE_MATLAB
#include <mex.h>
#define printf mexPrintf
#endif

typedef unsigned char uchar;
typedef unsigned long ulong;

#define BCV_ISNAN(x) ((x)!=(x)) //! check if nan
#define BCV_ODD(x) (((x) % 2) == 1) //! check if odd
#define BCV_EVEN(x) (((x) % 2) == 0) //! check if even 
#define BCV_SIGN(x) (((x) > 0) - ((x) < 0)) // -1, 0, or +1
enum { BCV_NORM_2 = 2, BCV_NORM_1 = 1, BCV_NORM_INF = 10 };



using namespace std;

void rgb2lab(vector<float>& out, const vector<float>& in);

void lab2rgb(vector<float>& out, const vector<float>& in);

void sobel(vector<float>& gx, vector<float>& gy, const vector<float>& in, int rows, int cols);

vector<float> get_exp_kernel1d(int d, float sigma);

int conv2d(vector<float>& out, const vector<float>& in, const vector<float>& K, 
        int rows, int cols, int rows_k, int cols_k);

int convc(vector<float>& out, const vector<float>& in, int rows, int cols, const vector<float>& K, int dim);

int blur(vector<float>& out, const vector<float>& in, int rows, int cols, const vector<float>& K);

void laplacian_of_gaussian(vector<float>& f, int size, float sigma);

void lmfilter(vector<float>& f, int size, float theta, float sx, float sy, int deriv_x, int deriv_y);

int gabor(vector<float>& gc, vector<float>& gs, int size, float theta, float scale);

void compute_integral_image(vector<float>& out, const vector<float>& in, int rows, int cols);

void bilateral_filter(vector<float>& out, const vector<float>& in, 
        int rows, int cols, float sigma_s, float sigma_r);

void bilateral_filter(vector<float>& out, const vector<float>& in, 
        int rows, int cols, float sigma_s, float sigma_r, int num_iters);

void nlmeans_filter(vector<float>& out, const vector<float>& in, 
        int rows, int cols, int filtsize, float sigma_r, int search_sz);

void nlmeans_filter_extractwindow(vector<float>& f, const vector<float>& img, 
        int filtsize, int r, int c, int rows, int cols, int chan);

float norm(const vector<float>& x, int type);

float dist(const vector<float>& x, const vector<float>& y, int type);

vector<vector<float> > get_filterbank(int num_theta, int num_log, int sz);

vector<vector<float> > apply_filterbank_to_image(const vector<float>& img, int rows, int cols, const vector<vector<float> >& filterbank, int filter_sz);

#endif // BCV_BASIC_H_
