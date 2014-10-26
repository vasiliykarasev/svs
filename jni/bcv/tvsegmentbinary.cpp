// @file tvsegmentbinary.cpp
#include "tvsegmentbinary.h"

tvsegmentbinary::tvsegmentbinary() { 
}

tvsegmentbinary::~tvsegmentbinary() {
}

tvsegmentbinary::tvsegmentbinary(tvsegmentbinary_params* p) { 
    sigma_u = 0.5f*1.0/sqrt(8.0f);
    sigma_y = 0.5f*1.0/sqrt(8.0f);
   
    // initialize optimization variables
    u = vector<float>(p->nnodes, 0); // nodes
    x = vector<float>(p->nnodes, 0);     
    y = vector<float>(p->nedges, 0); 
    
    vector<float> dx = vector<float>(p->nedges, 0); // nabla u
    vector<float> dxt = vector<float>(p->nnodes, 0); // div

    for (int t = 0; t < p->max_iters; ++t) {
        bcv_apply_sparse_op(dx, p->D, x, 'n');
        prox_tv_penalty(y, dx, p->weights, sigma_y);
        bcv_apply_sparse_op(dxt, p->D, y, 't');
        prox_unary_and_update(x, u, dxt, p->unary, sigma_u); 
    } 
}

void tvsegmentbinary::set_sigma(float sigma_u_, float sigma_y_) {
    sigma_u = sigma_u_;
    sigma_y = sigma_y_;
}

void tvsegmentbinary::prox_tv_penalty(vector<float>& y, 
    const vector<float>& dx, const vector<float>& weights, float sigma) { 
    for (size_t i = 0; i < y.size(); ++i) { 
        y[i] += dx[i]*sigma;
        y[i] = -BCV_SIGN(y[i])*max( -weights[i], -abs(y[i]) );
    }
}

void tvsegmentbinary::prox_unary_and_update(vector<float>& x, 
    vector<float>& u, const vector<float>& dxt, const vector<float>& unary, float sigma) { 
    float uprev;    
    for (size_t i = 0; i < u.size(); ++i) {
        uprev = u[i];
        u[i] = u[i] - sigma*(dxt[i] + unary[i]);
        u[i] = max(0.0f, u[i]);
        u[i] = min(1.0f, u[i]);
        x[i] = 2*u[i] - uprev;    
    }
}

#ifdef HAVE_SSE
void tvsegmentbinary::prox_tv_penalty_sse(vector<float>& y, 
    const vector<float>& dx, const vector<float>& weights, float sigma) { 
    assert( 0 && "fuck" );
}

void tvsegmentbinary::prox_unary_and_update_sse(vector<float>& x, 
    vector<float>& u, const vector<float>& dxt, const vector<float>& unary, float sigma) { 
    assert( 0 && "fuck" );
}
#endif


vector<uchar> tvsegmentbinary::get_segmentation() {
    vector<uchar> s;
    s.reserve( u.size() );
    for (size_t i = 0; i < u.size(); ++i) { s.push_back( (u[i] > 0.5f) ); }
    return s;
}

vector<float> tvsegmentbinary::get_result() {
    return u;
}
