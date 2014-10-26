// @file tvsegmentbinary.h
// @ref tvsegmentbinary.h is an implementation of primal-dual solver for a
// binary tv-regularized segmentation problem of the form:
//     $$ \min_u f^T u + | W D u |_1 $$
// where:
//      $u \in [0,1]^n$ is the optimization variable (foreground/background partition)
//      $f$ is the unary potential (f < 0 prefers u=1, f > 0 prefers u=0)
//      $W$ is the diagonal matrix of pairwise weights, and $D$ is the difference
//      operator. $| W D u |_1$ is the total variation regularizer.

#ifndef BCV_TVSEGMENTBINARY_H_
#define BCV_TVSEGMENTBINARY_H_

#include <cstdlib>
#include <cmath>
#include <limits>
#include "bcv_basic.h"
#include "bcv_sparse_op.h"

#ifdef HAVE_SSE
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

using namespace std;

//! tv segmentation parameters.
struct tvsegmentbinary_params {
    int nnodes;
    int nedges;
    int max_iters; //! maximum number of iterations
    vector<float> unary; //! unary term
    vector<float> weights; //! weights on edges
    bcv_sparse_op<int> D; //! difference operator on a graph
};

class tvsegmentbinary {
    public:
        //! main function that performs optimization
        tvsegmentbinary(tvsegmentbinary_params* p);
        ~tvsegmentbinary();
        tvsegmentbinary();
        vector<uchar> get_segmentation();
        vector<float> get_result();
        void set_sigma(float sigma_u_, float sigma_y_);
    private:
        float sigma_u;
        float sigma_y;
        vector<float> u; //! primal variable
        vector<float> x; 
        vector<float> y; 

        void prox_tv_penalty(vector<float>& y, const vector<float>& dx, 
            const vector<float>& weights, float sigma);
        void prox_unary_and_update(vector<float>& x, vector<float>& u, 
            const vector<float>& dxt, const vector<float>& unary, float sigma);
        
        #ifdef HAVE_SSE
        void prox_tv_penalty_sse(vector<float>& y, const vector<float>& dx, 
            const vector<float>& weights, float sigma);
        void prox_unary_and_update_sse(vector<float>& x, vector<float>& u, 
            const vector<float>& dxt, const vector<float>& unary, float sigma);
        #endif
};

#endif // BCV_TVSEGMENTBINARY_H_
