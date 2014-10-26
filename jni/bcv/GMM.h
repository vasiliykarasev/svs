//! @file GMM.h
#ifndef GMM_H_
#define GMM_H_

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>

using namespace std;

//! parameters of each component in the gaussian mixture
struct componentParam {
    vector<float> mu; //! 3x1 mean
    vector<float> cov; //! 3x3 covariance matrix
    vector<float> covinv; //! 3x3 inverse of the covariance
    float covdet; //! determinant
};

//! Class for gaussian mixture model, specific to grabcut. 
class GMM {
public:
    int numclusters; //! @var number of clusters.
    vector<componentParam> clusterParam; //! @var cluter parameters
    vector<float> clusterProb; //! @var mixture weights
    vector<int> assignment; //! @var vector of cluster assignments
    float regularization_val; //! @var eps in (C + eps I)
    vector<float> minval; //! @var minvalues in the image for each channel
    vector<float> maxval; //! @var maxvalues in the image for each channel
//-----------------------------------------------------------------------------
    GMM();
    GMM(int K, float reg_val=1e-2);

    //! initialize parameters of the gaussian mixture model
    void initParameters();
    //! get maximum/minimum value of the image for future scale estimation
    void initImageRange(const vector<float>& img);
    //! given an image, returns estimated cluster assignments    
    void getClusterAssignments(const vector<float>& img);
    //! update GMM parameters (given cluster assignments)
    void updateParameters(const vector<float>& img);
    //! Evaluate likelihood for a single LAB value (single pixel) 
    float evaluatePointLikelihood(const vector<float>& p);
    //! Evaluate sum-likelihood for the entire image
    float evaluateTotalLikelihood(const vector<float>& img);
    //! Evaluate likelihood for an entire image and return a vector
    vector<float> evaluateLikelihood(const vector<float>& img);

    void print_mixture_weights();
    void print_means();
    void print_covariances();
private:
    //! compute likelihood of the GMM components
    vector<float> computeClusterDistance(const vector<float>& p);    
    //! calculate argument of exponent in the Normal distribution
    float compute_exp_arg(const vector<float>& p, const vector<float>& mu,                        
        const vector<float>& covinv);
    //! calculate the inverse of the 3x3 covariance matrix
    float compute3x3symm_inverse( 
            const vector<float>& in, vector<float>& out);    

    void regularize_cov( vector<float>& m, float val);
    void ensure_posdef( vector<float>& m, float val);

    vector<float> init_mu();
    vector<float> init_mu(const vector<float>& img);
    vector<float> init_cov();
};

#endif //GMM_H_
