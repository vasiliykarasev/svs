//! @file bcv_basic.cpp
#include "bcv_basic.h"

//! Converts an RGB image to LAB.
//! input is assumed to be in range [0,1]. 3 channels are assumed, and are
//! assumed to be interleaved.
//! in-place operation is OK.
void rgb2lab(vector<float>& out, const vector<float>& in) {

    float x, y, z, fx, fy, fz, r, g, b;
    float Xn = 95.0f;
    float Yn = 100.0f;
    float Zn = 108.0f;
    int n = in.size()/3; 
    for (int i = 0; i < n; ++i) {
        r = in[3*i+0];
        g = in[3*i+1];
        b = in[3*i+2];

        if BCV_ISNAN(r) { r = 0; }
        if BCV_ISNAN(g) { g = 0; }
        if BCV_ISNAN(b) { b = 0; }

        // go from (R,G,B) to (R,G,B)linear
        if (r <= 0.04045f) { r = r / 12.92f; }
        else { r = pow( (double)(r + 0.055f)/(1.0f+0.055f), 2.4 ); }

        if (g <= 0.04045) { g = g / 12.92; }
        else { g = pow( (double)(g + 0.055f)/(1.0f+0.055f), 2.4 ); }
        if (b <= 0.04045) { b = b / 12.92; }
        else { b = pow( (double)(b + 0.055)/(1.0f+0.055f), 2.4 ); }

        // convert to xyz:
        x = 0.4124f*r + 0.3576f*g + 0.1805f*b;
        y = 0.2126f*r + 0.7152f*g + 0.0722f*b;
        z = 0.0193f*r + 0.1192f*g + 0.9505f*b;

        fx = x/Xn;
        fy = y/Yn;
        fz = z/Zn;

        if (fx > 0.0088564517f)
            fx = (float)pow( (double)fx, 1.0/3.0 );
        else
            fx = 7.787037037f * fx + 0.13793103448f;

        if (fy > 0.0088564517f)
            fy = (float)pow( (double)fy, 1.0/3.0 );
        else
            fy = 7.787037037f * fy + 0.13793103448f;

        if (fz > 0.0088564517f)
            fz = (float)pow( (double)fz, 1.0/3.0 );
        else
            fz = 7.787037037f * fz + 0.13793103448f;

        out[3*i+0] = 116.0f*fy - 16.0f;
        out[3*i+1] = 500.0f*(fx - fy);
        out[3*i+2] = 200.0f*(fy - fz);
    }
}

//! Converts an LAB image to RGB.
//! It is assumed that input and output have three interleaved channels,
//! and that input is in [0,1]. Output will be in [0,1]
//! in-place operation is OK.
void lab2rgb(vector<float>& out, const vector<float>& in) {
    float Xn = 95.0f;                                                            
    float Yn = 100.0f;                                                           
    float Zn = 108.0f;                                                           
    float fx, fy, fz, x, y, z, R, G, B, l, a, b;           

    int n = in.size()/3;
    for (int i = 0; i < n; ++i) { 
        l = in[3*i+0];
        a = in[3*i+1];
        b = in[3*i+2];
        if BCV_ISNAN(l) { l = 0; }
        if BCV_ISNAN(a) { a = 0; }
        if BCV_ISNAN(b) { b = 0; }        

        fy = 0.0086206896f*(l+16.0f);                              
        fx = fy + 0.002f*a;                                       
        fz = fy - 0.005f*b;                                       

        if (fx > 0.20689655172f)                                             
            fx = fx*fx*fx;                                                  
        else                                                                
            fx = 0.12841854934f*fx - 0.01771290335f;                          

        if (fy > 0.20689655172f)                                             
            fy = fy*fy*fy;                                                  
        else                                                                
            fy = 0.12841854934f*fy - 0.01771290335f;                          

        if (fz > 0.20689655172f)                                             
            fz = fz*fz*fz;                                                  
        else                                                                
            fz = 0.12841854934f*fz - 0.01771290335f;                          

        x = fx*Xn;                                               
        y = fy*Yn;                                               
        z = fz*Zn; 

        R = +3.2406f*x -1.5372f*y -0.4986f*z; 
        G = -0.9689f*x +1.8758f*y +0.0415f*z; 
        B = +0.0557f*x -0.2040f*y +1.0570f*z; 

        // go from (R,G,B)linear to (R,G,B)                                 
        if (R <= 0.0031308f) { R = R*12.92f; }
        else { R = (1.055)*pow( (double)R,1.0/2.4)-0.055; }

        if (G <= 0.0031308f) { G = G*12.92f; }
        else { G = (1.055)*pow( (double)G,1.0/2.4)-0.055; }

        if (B <= 0.0031308f) { B = B*12.92f; }
        else { B = (1.055)*pow( (double)B,1.0/2.4)-0.055; }

        out[3*i+0] = R;
        out[3*i+1] = G;
        out[3*i+2] = B;
    }
}

//! Convolution with a kernel K, which is centered at zero. Kernel needs to
//! have an odd number of entries.
//! @param[out] out - convolved image
//! @param[in] in  - input image
//! @param[in] rows, cols - size of the input image
//! @param[in] K - one-dimensional kernel
//! @param[in] dim - 0 - do along vertical, 1 - do along horizontal
int convc(vector<float>& out, const vector<float>& in, int rows, int cols, const vector<float>& K, int dim) {
    int d = K.size();
    int n = in.size();
    int chan = n/rows/cols;
    if ((int)out.size() != n) { out = vector<float>(n); }
    
    if BCV_EVEN(d) {
        printf("Kernel should have odd length!\n");
        printf("Since it is assumed that it is centered at the origin\n");
        return 1;
    }
    int center = d/2;
    int i;
    float f, rr, cc;

    if (chan > 1) { // for multichannel image (e.g. RGB). 
        if (dim == 0) { // do along vertical.
            for (int c = 0; c < cols; ++c) { 
                for (int r = 0; r < rows; ++r) {
                    for (int k = 0; k < chan; ++k) { 
                        f = 0.0;
                        for (int u = -center; u <= center; ++u) {
                            rr = min(rows-1, max(0, r-u) );
                            i = linear_index(rr, c, k, cols, chan);
                            f += K[center+u] * in[i];
                        }
                        i = linear_index(r, c, k, cols, chan);
                        out[i] = f;
                    }
                }
            }
        } else { // do along horizontal
            for (int r = 0; r < rows; ++r) { 
                for (int c = 0; c < cols; ++c) {
                    for (int k = 0; k < chan; ++k) { 
                        f = 0.0;
                        for (int u = -center; u <= center; ++u) {
                            cc= min(cols-1, max(0, c-u) );
                            i = linear_index(r, cc, k, cols, chan);
                            f += K[center+u] * in[i];
                        }
                        i = linear_index(r, c, k, cols, chan);
                        out[i] = f;
                    }
                }
            }
        }
    } else {
        // separate block for grayscale image. this is ugly, but allows us to
        // skip MANY checks in the forloop, and seems to shave a few ms.
        if (dim == 0) { // do along vertical.
            for (int c = 0; c < cols; ++c) { 
                for (int r = 0; r < rows; ++r) {
                    f = 0.0;
                    for (int u = -center; u <= center; ++u) {
                        rr = min(rows-1, max(0, r-u) );
                        i = linear_index(rr, c, cols);
                        f += K[center+u] * in[i];
                    }
                    i = linear_index(r, c, cols);
                    out[i] = f;
                }
            }
        } else { // do along horizontal
            for (int r = 0; r < rows; ++r) { 
                for (int c = 0; c < cols; ++c) {
                    f = 0.0;
                    for (int u = -center; u <= center; ++u) {
                        cc= min(cols-1, max(0, c-u) );
                        i = linear_index(r, cc, cols);
                        f += K[center+u] * in[i];
                    }
                    i = linear_index(r, c, cols);
                    out[i] = f;
                }
            }
        }
    }
    return 0;
}

//! 2D convolution with a (presumably nonseparable) kernel K.
//! @param[out] out - output image.
//! @param[in] in - input image (dimensions rows, cols)
//! @param[in] K - input kernel (dimensions rows_k, cols_k)
//! no padding is performed.
int conv2d(vector<float>& out, const vector<float>& in, const vector<float>& K, 
        int rows, int cols, int rows_k, int cols_k) {
    int d = K.size();
    int n = in.size();
    int chan = n/rows/cols;
    int kr, kc;
    if ((int)out.size() != n) { out = vector<float>(n); }
   
    int offset_rows = rows_k/2+1;
    int offset_cols = cols_k/2+1;
    // this is more of a projection than standard convolution.
    for (int r = offset_rows; r < rows-offset_rows; ++r) { 
        for (int c = offset_cols; c < cols-offset_cols; ++c) { 
            for (int k = 0; k < chan; ++k) {
                // do convolution:
                float val = 0.0f;
                for (int rr = r-offset_rows, kr=0; kr < rows_k; ++rr, ++kr) { 
                    for (int cc = c-offset_cols, kc=0; kc < cols_k; ++cc, ++kc) { 
                        int id1 = linear_index(rr, cc, k, cols, chan);
                        int id2 = linear_index(kr, kc, cols_k);
                        val += in[id1]*K[id2];
                    }
                }
                int id = linear_index(r, c, k, cols, chan);
                out[id] = val;
            }
        }
    }
    // handle corners:
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if ((c >= offset_cols) && (c < cols-offset_cols) && 
                    (r >= offset_rows) && (r < rows-offset_rows)) { 
                continue; // skip the middle portion
            }
            for (int k = 0; k < chan; ++k) { 
                // do convolution:
                float val = 0.0f;
                for (int rr = r-offset_rows, kr=0; kr < rows_k; ++rr, ++kr) { 
                    if ((rr < 0) || (rr >= rows)) { continue; }
                    for (int cc = c-offset_cols, kc=0; kc < cols_k; ++cc, ++kc) { 
                        if ((cc < 0) || (cc >= cols)) { continue; }
                        int id1 = linear_index(rr, cc, k, cols, chan);
                        int id2 = linear_index(kr, kc, cols_k);
                        val += in[id1]*K[id2];
                    }
                }
                int id = linear_index(r, c, k, cols, chan);
                out[id] = val;
            }
        }
    }

    return 0;
}

//! Returns a filter used in Leung-Malik 2001. Filter is normalized to be zero
//! mean and to have unity L1 norm.
//! @param[out] f - output filter.
//! @param[in] size - size of the filter (filter is square). must be odd.
//! @param[in] sx, sy - variance along x and y.
//! @param[in] deriv_x, deriv_y - order of the derivative (0,1,2) along x and y.
void lmfilter(vector<float>& f, int size, float theta, float sx, float sy, int deriv_x, int deriv_y) {
    if (BCV_EVEN(size)) { 
        printf("Error. filter size should be odd.\n");
        return;
    }
    // TODO: check variance positive
    f = vector<float>(size*size);
    int h = (size-1)/2;
    float Zx = 1.0f/sqrt(M_PI*sx);
    float Zy = 1.0f/sqrt(M_PI*sy);
    
    for (int x0 = -h; x0 <= h; ++x0) { 
        for (int y0 = -h; y0 <= h; ++y0) { 
            float x = +x0*cos(theta) + y0*sin(theta);            
            float y = -x0*sin(theta) + y0*cos(theta);            
           
            float gx = exp(-0.5f*x*x/sx)*Zx;
            float gy = exp(-0.5f*y*y/sy)*Zy;
            if (deriv_x == 1) { gx = -gx*(x/sx); }
            else if (deriv_x == 2) { gx = +gx*((x*x-sx)/sx/sx); }
            if (deriv_y == 1) { gy = -gy*(y/sy); }
            else if (deriv_y == 2) { gy = +gy*((y*y-sy)/sy/sy); }

            int idx = linear_index(h+y0, h+x0, size);
            f[idx] = gx*gy;
        }
    }
    // normalization step. subtract mean, and make sure L1 norm is 1.
    float mu = accumulate(f.begin(), f.end(), 0.0f)/float(f.size());
    transform(f.begin(), f.end(), f.begin(), bind1st(plus<float>(), -mu));
    float l1 = norm(f, BCV_NORM_1);
    transform(f.begin(), f.end(), f.begin(), bind1st(multiplies<float>(), 1.0f/l1));
}

//! Returns Laplacian of Gaussian filter. Normalized to 0 mean, unity L1 norm.
//! @param[out] f - output filter.
//! @param[in] size - size of the filter (filter is square). must be odd.
//! @param[in] sigma - variance
void laplacian_of_gaussian(vector<float>& f, int size, float sigma) {
    if (BCV_EVEN(size)) { 
        printf("Error. filter size should be odd.\n");
        return;
    }
    // TODO: check variance positive
    f = vector<float>(size*size);
    int h = (size-1)/2;
    float Z = 1.0f/(2.0f*sigma);
    for (int x0 = -h; x0 <= h; ++x0) { 
        for (int y0 = -h; y0 <= h; ++y0) { 
            float val = (x0*x0 + y0*y0)*Z;
            int idx = linear_index(h+y0, h+x0, size);
            f[idx] = (val - 1.0f)*exp(-val);
        }
    } 
    // normalization step. subtract mean, and make sure L1 norm is 1.
    float mu = accumulate(f.begin(), f.end(), 0.0f)/float(f.size());
    transform(f.begin(), f.end(), f.begin(), bind1st(plus<float>(), -mu));
    float l1 = norm(f, BCV_NORM_1);
    transform(f.begin(), f.end(), f.begin(), bind1st(multiplies<float>(), 1.0f/l1));
}


//! Returns a quadrature pair of gabor wavelets, with a certain scale and orientation.
//! @param[out] gc, gs - output 2D wavelets (cos,sin) centered at the middle
//! @param[in] size - size of the wavelet (which is square)
//! @param[in] theta - [0,pi) orientation of the wavelet.
//! @param[in] scale - [1,\inf) controls the precision (inverse of variance).
//! When set to 1, it roughly coincides with the scale used in the paper.
//!     This function is a direct matlab port of Active Basis code seen below:
//!         http://www.stat.ucla.edu/~ywu/AB/ActiveBasisMarkII.html
int gabor(vector<float>& gc, vector<float>& gs, int size, float theta, float scale) {
    int expand = 12;
    int h = floor(size/2);
    int hsq = h*h;

    gc = vector<float>( 10*(2*h+1)*(2*h+1), 0.0f);
    gs = vector<float>( 10*(2*h+1)*(2*h+1), 0.0f);

    vector<float> gauss = vector<float>( (2*h+1)*(2*h+1), 0.0f);
    float precision = 0.01f*scale;
    for (int x0 = -h; x0 <= h; ++x0) { 
        for (int y0 = -h; y0 <= h; ++y0) { 
            if (x0*x0+y0*y0 > hsq) { continue; }
            float x = (+x0*cos(theta) + y0*sin(theta))*expand/h;            
            float y = (-x0*sin(theta) + y0*cos(theta))*expand/h;            
            float val = exp(-(4.0f*x*x+y*y)*precision);
            int idx = linear_index(h+y0+1, h+x0+1, 2*h+1);
            gauss[idx] = val;
            gc[idx] = val*cos(x);
            gs[idx] = val*sin(x);
        }
    }
    // normalize:
    /*
    //float gauss_sum = std::accumulate(gauss.begin(), gauss.end(), 0.0f);
    //float gc_sum = std::accumulate(gc.begin(), gc.end(), 0.0f);
    //float r = gc_sum/gauss_sum;
    // remove dc
    float energy_cos = 0.0f;
    float energy_sin = 0.0f;
    for (int i = 0; i < gc.size(); ++i) { 
//    gc[i] = gc[i] - gauss[i]*r;
energy_cos += gc[i]*gc[i]; 
energy_sin += gs[i]*gs[i]; 
}
std::transform(gc.begin(), gc.end(), gc.begin(), 
std::bind1st(std::multiplies<float>(),1.0f/sqrt(energy_cos)) ); 
std::transform(gs.begin(), gs.end(), gs.begin(), 
std::bind1st(std::multiplies<float>(),1.0f/sqrt(energy_sin)) ); 
     */
return 0;
}

//! Computes an integral image.
void compute_integral_image(vector<float>& out, 
        const vector<float>& in, int rows, int cols) {
    int chan = in.size()/rows/cols;
    out = vector<float>(rows*cols*chan, 0.0f);
    // take care of the left column
    int c = 0;
    for (int r = 1; r < rows; ++r) {
        for (int k = 0; k < chan; ++k) { 
            int idx = linear_index(r, c, k, cols, chan);
            int idx_prev = linear_index(r-1, c, k, cols, chan);
            out[idx] = out[idx_prev]+in[idx];
        }
    }
    // take care of the top row
    int r = 0;
    for (int c = 1; c < cols; ++c) { 
        for (int k = 0; k < chan; ++k) { 
            int idx = linear_index(r, c, k, cols, chan);
            int idx_prev = linear_index(r, c-1, k, cols, chan);
            out[idx] = out[idx_prev]+in[idx];
        }
    }
    // compute the rest of the integral image
    for (int r = 1; r < rows; ++r) {  
        for (int c = 1; c < cols; ++c) { 
            for (int k = 0; k < chan; ++k) { 
                int idx = linear_index(r, c, k, cols, chan);
                int idx_leftup = linear_index(r-1, c-1, k, cols, chan);
                int idx_left = linear_index(r, c-1, k, cols, chan);
                int idx_up = linear_index(r-1, c, k, cols, chan);
                out[idx] = in[idx] + out[idx_left] + out[idx_up] - out[idx_leftup];
            }
        }
    }
}


//! Performs convolution with a symmetric kernel K (with odd number of coefficients)
//! along both horizontal and vertical directions.
int blur(vector<float>& out, const vector<float>& in, int rows, int cols, const vector<float>& K) {
    int n = rows*cols;
    int sz = in.size();

    if ((int)out.size() != sz) { out = vector<float>(sz); }

    if BCV_EVEN((int)K.size()) {
        printf("Kernel should have odd length!\n");
        printf("Since it is assumed that it is centered at the origin\n");
        return 1;
    }

    vector<float> temp = vector<float>(n); 
    convc(temp, in, rows, cols, K, 0);  
    convc(out, temp, rows, cols, K, 1);  
    return 0;
}


//! Performs Sobel operation. Input can be a multichannel image.
void sobel(vector<float>& gx, vector<float>& gy, const vector<float>& in, int rows, int cols) {
    float f_11, f_12, f_13, f_21, f_22, f_23, f_31, f_32, f_33;
    int i;

    int n = rows*cols;
    int sz = in.size();
    int chan = sz/n;

    if (gx.size() != sz) { gx = vector<float>(sz); }
    if (gy.size() != sz) { gy = vector<float>(sz); }

    for (int c = 0; c < cols; ++c) { 
        for (int r = 0; r < rows; ++r) { 
            for (int k = 0; k < chan; ++k) { 
                i = linear_index(r,c,k, cols, chan);
                f_22 = in[i];
                // zero padding.
                // f_11 = f_12 = f_13 = f_21 = f_23 = f_31 = f_32 = f_33 = 0;
                // same as center:
                f_11 = f_12 = f_13 = f_21 = f_23 = f_31 = f_32 = f_33 = f_22;
                if (c < cols-1) {
                    f_23 = in[ linear_index(r,c+1,k, cols, chan) ];
                    if (r < rows-1) {
                        f_33 = in[ linear_index(r+1,c+1,k, cols, chan) ];
                    }
                    if (r > 0) { 
                        f_13 = in[ linear_index(r-1,c+1,k, cols, chan) ];
                    }
                }
                if (c > 0) { 
                    f_21 = in[ linear_index(r,c-1,k, cols, chan) ];
                    if (r < rows-1) {
                        f_31 = in[linear_index(r+1,c-1,k, cols, chan)];
                    }
                    if (r > 0) { 
                        f_11 = in[linear_index(r-1,c-1,k, cols, chan)];
                    }
                }
                gx[i] = 
                    f_33*(-1.0) + f_23*(-2.0) + f_13*(-1.0) + 
                    f_32*(+0.0) + f_22*(+0.0) + f_12*(+0.0) + 
                    f_31*(+1.0) + f_21*(+2.0) + f_11*(+1.0);

                gy[i] = 
                    f_33*(-1.0) + f_23*(+0.0) + f_13*(+1.0) + 
                    f_32*(-2.0) + f_22*(+0.0) + f_12*(+2.0) + 
                    f_31*(-1.0) + f_21*(+0.0) + f_11*(+1.0);
            }
        }
    }
}


/*
   int inline getchan(int i, int rows, int cols, int chan) { 
   return i / chan;
   }
   int inline getrow(int i, int rows, int cols, int chan) {
   return i / cols / chan;
   }

   int inline getcol(int i, int rows, int cols, int chan) {
   return i - (i/cols)*cols;
   }*/

//! Compute norm of the input vector.
float norm(const vector<float>& x, int type=BCV_NORM_2) {
    int n = x.size();
    float d = 0.0f;
    if (type == BCV_NORM_1) { 
        for (int i = 0; i < n; ++i) {
            d += abs(x[i]);
        }
    } else if (type == BCV_NORM_2) { 
        for (int i = 0; i < n; ++i) { 
            d += x[i]*x[i];
        }
        d = sqrt(d);
    } else if (type == BCV_NORM_INF) { 
        for (int i = 0; i < n; ++i) {
            d = max(d, abs(x[i]));
        }
    } else {
        printf("Unrecognized norm type.\n");
    }
    return d;
}

//! Computes distance between two vectors
float dist(const vector<float>& x, const vector<float>& y, int type=BCV_NORM_2) {
    int n = x.size();
    int m = y.size();
    if (m !=n) { 
        printf("dist: Vector sizes are not equal!\n");
        return 0;
    }
    float d = 0.0;
    if (type == BCV_NORM_1) { 
        for (int i = 0; i < n; ++i) {
            d += abs(x[i]-y[i]);
        }
    } else if (type == BCV_NORM_2) { 
        for (int i = 0; i < n; ++i) { 
            d += (x[i]-y[i])*(x[i]-y[i]);
        }
        d = sqrt(d);
    } else if (type == BCV_NORM_INF) { 
        for (int i = 0; i < n; ++i) {
            d = max(d, abs(x[i]-y[i]));
        }
    } else {
        printf("Unrecognized norm type.\n");
    }
    return d;
}

//! maximum value is 1 (does not compute the prob.distribution).
vector<float> get_exp_kernel1d(int d, float sigma) {
    if BCV_EVEN(d) {
        printf("Increasing kernel size to make odd.\n");
        d = d + 1;
    }
    vector<float> kernel = vector<float>(d);
    int center = d/2;
    float s = 0.0;
    for (int i = 0; i < d; ++i) { 
        kernel[i] = exp(-(i-center)*(i-center)/sigma/sigma/2.0);
        s += kernel[i];
    }
    for (int i = 0; i < d; ++i) { kernel[i] /= s; }

    return kernel;
}

//! Performs bilateral filtering iteration.
//! @param[out] out - output image
//! @param[in] in - input image (dimensions rows, cols)
//! @param[in] sigma_s - location smoothin size (std.deviation)
//! @param[in] sigma_r - intensity smoothing size (std deviaiton)
//! input can be a multichannel image.
void bilateral_filter(vector<float>& out, const vector<float>& in, 
        int rows, int cols, float sigma_s, float sigma_r) {
    int n = rows*cols;
    int sz = in.size();
    int chan = sz/n;
    int ylo, yhi, xlo, xhi, dg;
    float d;
    float inv_sigma_r_sq = .5/(sigma_r*sigma_r);
    float inv_sigma_s_sq = .5/(sigma_s*sigma_s);
    float W, Z;
    int i_og, i;
    int filtsize = ceil(3*sigma_s);
    if BCV_EVEN(filtsize) { filtsize++; }

    if (out.size() != sz) { out = vector<float>(sz, 0); }

    float EXP_MAX_VAL = 5.0;
    int EXP_NUM = 256;
    vector<float> exp_ = vector<float>(EXP_NUM);
    for (int i = 0; i < exp_.size(); ++i) {
        float v = float(i)/float(exp_.size())*EXP_MAX_VAL;
        exp_[i] = exp(-v);
    }
    for (int r = 0; r < rows; ++r) { 
        ylo = max(0, r-filtsize/2);
        yhi = min(rows-1, r+filtsize/2);
        for (int c = 0; c < cols; ++c) { 
            xlo = max(0, c-filtsize/2);
            xhi = min(cols-1, c+filtsize/2);

            Z = 1e-10;
            i_og = linear_index(r, c, 0, cols, chan);
            for (int xs=xlo; xs<=xhi; ++xs) {
                for (int ys=ylo; ys<=yhi; ++ys) {
                    // compute image distance...
                    i = linear_index(ys, xs, 0, cols, chan);
                    d = 0.0;
                    for (int k = 0; k < chan; ++k) { 
                        d += (in[i+k]-in[i_og+k])*(in[i+k]-in[i_og+k]);
                    }
                    dg = (xs-c)*(xs-c)+(ys-r)*(ys-r);
                    float val = d*inv_sigma_r_sq + float(dg)*inv_sigma_s_sq;
                    W = exp_[ min(EXP_NUM-1, int(EXP_NUM*val/EXP_MAX_VAL)) ];
                    Z += W;
                    for (int k = 0; k < chan; ++k) { 
                        out[i_og+k] += W*in[i+k]; 
                    }
                }
            }
            // normalize result
            for (int k = 0; k < chan; ++k) { 
                out[i_og+k] /= Z;
            }
        }
    }
}

//! Performs multiple bilateral filtering iterations (see bilateral_filter fn)
void bilateral_filter(vector<float>& out, const vector<float>& in, 
        int rows, int cols, float sigma_s, float sigma_r, int num_iters) {
    if (num_iters == 1) {
        bilateral_filter(out, in, rows, cols, sigma_s, sigma_r);
    } else {
        int n = in.size();
        vector<float> temp = in;
        for (int i = 0; i < num_iters; ++i) { 
            bilateral_filter(out, temp, rows, cols, sigma_s, sigma_r);
            if (i < num_iters-1) { temp = out; } // prepare for next iterations
        }
    }
}

//! Performs nonlocal means filtering
//! @param[out] out - output image
//! @param[in] in - input image (dimensions rows, cols)
//! @param[in] sigma_r - width of the intensity kernel
//! @param[in] filtsize - size of the window
//! @param[in] search_sz - size of the neighborhood over which search is done
//! input can be a multichannel image.
void nlmeans_filter(vector<float>& out, const vector<float>& in, 
        int rows, int cols, int filtsize, float sigma_r, int search_sz) {
    int n = rows*cols;
    int sz = in.size();
    int chan = sz/n;

    int ylo, yhi, xlo, xhi, dg;
    float d, W, Z;
    int i_og, i;

    float inv_sigma_r_sq = .5/(sigma_r*sigma_r);
    filtsize = max( 3, filtsize );
    if BCV_EVEN(filtsize) { filtsize++; }
    float inv_filtsz = 1.0f/(filtsize*filtsize*chan);
    if BCV_EVEN(search_sz) { search_sz++; }


    if (out.size() != sz) { out = vector<float>(sz, 0); }

    float EXP_MAX_VAL = 5.0;
    int EXP_NUM = 256;
    vector<float> exp_ = vector<float>(EXP_NUM);
    for (int i = 0; i < exp_.size(); ++i) {
        float v = float(i)/float(exp_.size())*EXP_MAX_VAL;
        exp_[i] = exp(-v);
    }

    vector<float> w1 = vector<float>(filtsize*filtsize*chan);
    vector<float> w2 = vector<float>(filtsize*filtsize*chan);

    for (int r = 0; r < rows; ++r) { 
        ylo = max(0, r-search_sz/2);
        yhi = min(rows-1, r+search_sz/2);
        for (int c = 0; c < cols; ++c) { 
            xlo = max(0, c-search_sz/2);
            xhi = min(cols-1, c+search_sz/2);
            nlmeans_filter_extractwindow(w1, in, filtsize, r, c, rows, cols, chan);
            Z = 1e-10;
            i_og = linear_index(r, c, 0, cols, chan);
            for (int k = 0; k < chan; ++k) { out[i_og+k] = 0; }
            // filter
            for (int xs=xlo; xs<=xhi; ++xs) {
                for (int ys=ylo; ys<=yhi; ++ys) {
                    // compute image distance...
                    nlmeans_filter_extractwindow(w2, in, filtsize, ys, xs, rows, cols, chan); 
                    d = dist(w1, w2)*inv_filtsz;
                     
                    i = linear_index(ys, xs, 0, cols, chan);
                    float val = d*d*inv_sigma_r_sq;
                    W = exp_[ min(EXP_NUM-1, int(EXP_NUM*val/EXP_MAX_VAL)) ];
                    Z += W;
                    for (int k = 0; k < chan; ++k) { 
                        out[i_og+k] += W*in[i+k]; 
                    }
                }
            }
            // normalize result
            for (int k = 0; k < chan; ++k) { 
                out[i_og+k] /= Z;
            }
        }
    }
}

void nlmeans_filter_extractwindow(vector<float>& f, const vector<float>& img, 
        int filtsize, int r, int c, int rows, int cols, int chan) {
    // it is assumed that f.size() = filtsize*filtsize*chan
    int i = 0;
    for (int fr = -filtsize/2; fr <= filtsize/2; ++fr) { 
        int rr = min(max(r + fr, 0), rows-1);
        for (int fc = -filtsize/2; fc <= filtsize/2; ++fc) { 
            int cc = min(max(c + fc, 0), cols-1);
            for (int k = 0; k < chan; ++k) { 
                f[i] = img[ linear_index(rr, cc, k, cols, chan) ];
                i++; 
            }
        }
    }
}

//! Returns a collection of filters (with Leung-Malik filters)
//! @param[in] num_theta - number of orientation
//! @param[in] num_LoG - number of laplacian of gaussian scales
//! @param[in] sz - size of the filter (which is square)
vector<vector<float> > get_filterbank(int num_theta, int num_log, int sz) { 

    int num_filters = num_log + num_theta;
    vector<float> f;
    vector< vector<float> > filterbank;
    filterbank.reserve(num_filters);

    int deriv_x = 0;
    int deriv_y = 2;
    float sx = 3.0f*3.0f;
    float sy = 1.0f;
    float scale = 1.0f;
    for (int k = 0; k < num_theta; ++k) { 
        float theta = float(k)/float(num_theta)*M_PI;
        lmfilter(f, sz, theta, sx, sy, deriv_x, deriv_y);
        filterbank.push_back(f);        
    }
    for (int s = 0; s < num_log; ++s) { 
        laplacian_of_gaussian(f, sz, s+1.0);
        filterbank.push_back(f);
    }
    return filterbank;
}

//! Returns a vector of images filtered with filterbank.
//! @param[in] img, rows, cols - input image
//! @param[in] filterbank - vector of filters
//! @param[in] filter_sz - size of all filters
vector<vector<float> > apply_filterbank_to_image(const vector<float>& img, 
        int rows, int cols, const vector<vector<float> >& filterbank, int filter_sz) {
    vector<vector<float> > out_img;
    out_img.reserve( filterbank.size() );
    vector<float> temp;
    for (int k = 0; k < filterbank.size(); ++k) { 
        conv2d(temp, img, filterbank[k], rows, cols, filter_sz, filter_sz);
        out_img.push_back(temp);
    }
    return out_img;
}


//! adapted from rosetta code.
//! returns an interleaved vector of xy points (x0,y0, x1,y1, ...)
vector<int> bresenham_line(int x0, int y0, int x1, int y1) {
    // this should be an upper bound on how many points we need    
    vector<int> pts = vector<int>();
    int n = abs(x1-x0) + abs(y1-y0);
    pts.reserve(2*n);

    int dx = abs(x1-x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1-y0), sy = y0 < y1 ? 1 : -1; 
    int err = (dx>dy ? dx : -dy)/2, e2;
 
    for(;;) {
        pts.push_back(x0);
        pts.push_back(y0);
        if (x0==x1 && y0==y1) { break; }
        e2 = err;
        if (e2 >-dx) { err -= dy; x0 += sx; }
        if (e2 < dy) { err += dx; y0 += sy; }
    }
    return pts;
}