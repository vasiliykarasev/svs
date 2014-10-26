//! @file GMM.cpp
#include "GMM.h"
GMM::GMM() {
    maxval.clear();
    minval.clear();
    assignment.clear();
    clusterProb.clear();
    clusterParam.clear();
}
//-----------------------------------------------------------------------------

/*! Standard constructor for the Gaussian mixture model. Tries to initialize
    as many fields as possible. Parameters of the mixture (i.e. weights,
    means, and covariances are initalized to default values with DEFAULT
    SCALE -- meaning it is assumed that the image is normalized to [0,255]).
    The functions described below make it possible to estimate a more scale
    initialization scale.
    @param K - number of clusters in a GMM
    @param reg_val - a regularization parameter for the covariance matrix
    estimation and inverse of the covariance matrix. To avoid zero or worse
    yet negative determinants, and incorrect estimates of the covariance
    matrices, we regularize the matrix by adding a small value along the
    diagonal: \f$C \doteq C + \epsilon I\f$. reg_val describes this
    \f$\epsilon\f$. */
GMM::GMM(int K, float reg_val) {
    maxval = vector<float>(3, 255.0);
    minval = vector<float>(3, 0.0);
    assignment.clear();
    clusterProb.clear();
    clusterParam.clear();

    regularization_val = reg_val;
    numclusters = K;
    clusterParam = vector<componentParam>(K);
    // init parameters (but mind that here scaling is not intelligent) 
    initParameters();
}

/*! Initializes parameters of the GMM; mixture weights are initialized
    to be uniform. Means are chosen randomly in [0,255] (or whatever scale
    is estimated). Covariance matrices are initialized to be diagonal, with
    variance 255*255. */
void GMM::initParameters() {
    clusterProb = vector<float>(numclusters, 1.0/((float)numclusters) );
    for (int i = 0; i < numclusters; ++i) {
        clusterParam.at(i).mu = init_mu();
        // initialize space for the covariance matrix
        clusterParam.at(i).cov = vector<float>(9, 0.0f);
        clusterParam.at(i).covinv = vector<float>(9, 0.0f);

        clusterParam.at(i).cov = init_cov();
        clusterParam.at(i).covdet = 
            compute3x3symm_inverse( 
                    clusterParam.at(i).cov, clusterParam.at(i).covinv );
    }
}

/*! Given data (an image) and currently estimated parameters, find cluster 
    assignments.
    @param img - image data. this is assumed to be three channel data. */
void GMM::getClusterAssignments(const vector<float>& img) {
    int n = img.size() / 3;
    if (assignment.size() != n)
        assignment = vector<int>(n, 0);

    // for each pixel estimate the assignment.
    vector<float> pix(3, 0.0f);
    vector<float> d;
    for (size_t i = 0; i < n; ++i) {
        pix[0] = img[3*i];
        pix[1] = img[3*i+1];
        pix[2] = img[3*i+2];
        d = computeClusterDistance( pix ); 
        assignment[i] = min_element(d.begin(), d.end()) - d.begin();
    }
}

/*! Computes an inverse of the covariance matrix. It is assumed that the
    matrix is symmetric.
    @param in - input 3x3 matrix.
    @param out - output 3x3 matrix. It MUST be already initialized.
    @return det - determinant of input matrix.
 */
float GMM::compute3x3symm_inverse(const vector<float>& in, vector<float>& out) {
    float A,B,C,D,E,F,G,H,I;
    float a,b,c,d,e,f,g,h,i;
    a = in[0];
    d = in[1];
    g = in[2];
    b = in[3];
    e = in[4];
    h = in[5];
    c = in[6];
    f = in[7];
    i = in[8];

    A = e*i-f*h;
    B =-(d*i-f*g);
    C = d*h-e*g;
    D = -(b*i-c*h);
    E = (a*i-c*g);
    F =-(a*h-b*g);
    G = b*f-c*e;
    H = -(a*f-c*d);
    I = a*e-b*d;
    
    float det = a*(e*i-f*h) - b*(i*d-f*g) + c*(d*h-e*g);
    if (det < 0) {
        cout << "ERROR IN DETERMINANT CALCULATION!" << endl;
    }
    out[0] = A / det;
    out[1] = B / det;
    out[2] = C / det;
    out[3] = D / det;
    out[4] = E / det;
    out[5] = F / det;
    out[6] = G / det;
    out[7] = H / det;
    out[8] = I / det;
    return (float)det;
}

/*! Evaluate the total likelihood of the data under the current GMM model.
    This function sums individual likelihoods (i.e. for each data-point --
    for each pixel in the image), and returns the total value. 

    @note It is useful if you wish to estimate how well the GMM parameters
    are estimated, or if you wish to fit multiple models and keep the best
    one (neither are currently implemented). 
    @param img - image data
    @return likelihood of the model */
float GMM::evaluateTotalLikelihood(const vector<float>& img) {
    int n = img.size() / 3;
    float L=0;
    vector<float> pix = vector<float>(3, 0.0f);
    for (int i = 0; i<n; ++i) {
        pix[0] = img[3*i];
        pix[1] = img[3*i+1];
        pix[2] = img[3*i+2];
        L += evaluatePointLikelihood( pix );
    }
    return L;
}

/*! Evaluate likelihood under the current GMM model. This function returns
    likelihoods for every data point as a vector. Presumably combinining
    this computation in a single function would have lead to speedups, yet
    it turned out to be false.
    @note This function is not currently used.*/
vector<float> GMM::evaluateLikelihood(const vector<float>& img) {
    float f, Q;
    float TWOMPI3 = (2*M_PI) * (2*M_PI) * (2*M_PI);
    int n = img.size() / 3;
    vector<float> L(n, 0.0f);
    vector<float> Z(numclusters);
    vector<float> pix = vector<float>(3, 0.0f);
    for (int k = 0; k<numclusters; ++k) {
        Z[k] = clusterProb[k] / sqrt(TWOMPI3 * clusterParam[k].covdet);
    }

    for (int i = 0; i < n; ++i) {
        L[i] = 0;
        pix[0] = img[3*i];
        pix[1] = img[3*i+1];
        pix[2] = img[3*i+2];
        for (int k = 0; k < numclusters; ++k) {
            Q = compute_exp_arg( 
                    pix, clusterParam[k].mu, clusterParam[k].covinv);
            L[i] += exp(-0.5*Q) * Z[k];
        }
    }
    return L;
}

/*! Evaluate likelihood of a single given data point.
    @param z - a single LAB value
    @return likelihood
*/
float GMM::evaluatePointLikelihood(const vector<float>& z) {
    float L=0;
    float f, Q;
    float TWOMPI3 = (2*M_PI) * (2*M_PI) * (2*M_PI);
    for (int i = 0; i < numclusters; ++i) {
        Q = compute_exp_arg( 
                z, clusterParam[i].mu, clusterParam[i].covinv );
        f = exp(-0.5*Q) / sqrt(TWOMPI3 *clusterParam[i].covdet);
        L += clusterProb[i] * f;
    }
    return L;
}

/*! For a given point, calculate "distance" to each GMM cluster, given
    current estimates of the GMM parameters. 
    @param z - input LAB value
    @return d - vector of distances to the clusters
*/
vector<float> GMM::computeClusterDistance(const vector<float>& z) {
    vector<float> d(numclusters, 0);    
    float Q;
    // this function computes the following:
    // d = -log(pi) + 0.5*log(det(cov)) + 0.5*(z-mu)' cov^{-1} (z-mu)
    for (int i = 0; i < numclusters; ++i) {
        Q = compute_exp_arg( z, clusterParam[i].mu, clusterParam[i].covinv );
//        d[i] = -log( clusterProb[i] + 1e-6 ) + 
//            0.5*log(clusterParam[i].covdet) + 0.5*Q;
        d[i] = 0.5*log(clusterParam[i].covdet) + 0.5*Q;        
    }
    return d;
}

/*! update parameters of the GMM components, given current cluster
    assignments. */
void GMM::updateParameters(const vector<float>& img) {
    int k;
    int n = assignment.size();
    vector<int> nums(numclusters,0);
    
    vector<vector<float> > muest(numclusters, vector<float>(3,0.0f));
    vector<vector<float> > covest = vector<vector<float> >(numclusters);
    // set to zero
    for (int k=0; k<numclusters;++k) {
        covest[k] = vector<float>(9, 0.0f);
    }
    // compute \sum_{k \in C(k)} x_k  and \sum_{k \in C(k)} x_k x_k^T  
    for (size_t i = 0; i < n; ++i) {
        k = assignment[i]; // cluster index.
        nums.at(k) += 1;
        for (size_t u = 0; u < 3; ++u) {
            muest[k][u] += img[3*i+u];
        }
        for (int u1=0; u1<3; ++u1) { 
            for (int u2=0; u2<3; ++u2) {
                covest.at(k)[u1*3+u2] += img[3*i+u1]*img[3*i+u2];
            }
        }
    }
    // now compute: 1/(n-1) \sum x_i x_i^T - n \mu \mu^T
    // x_i x_i^T is stored in covest.
    // n\mu is stored in muest.
    for (size_t k = 0; k < numclusters; ++k) {
        int N = nums.at(k);
        if (N > 0) {
            // sample mean estimate.
            for (size_t u = 0; u < 3; ++u)
                muest[k][u] = muest[k][u] / float(N);
            // sample covariance estimate.
            for (int u1=0; u1 < 3; ++u1) { 
                for (int u2=0; u2 < 3; ++u2) {
                    covest[k][3*u1+u2] = 
                        1.0/(float(N))*covest[k][3*u1+u2] - 
                        muest[k][u1]*muest[k][u2];
                }
            }
        } else {
            //muest[k] = init_mu();
            muest[k] = init_mu(img);
            covest[k] = init_cov();
        }
    } 
    int sum = accumulate( nums.begin(), nums.end(), 0 );

    for (size_t k = 0; k < numclusters; ++k) {
        clusterParam.at(k).mu = muest[k];
        clusterParam.at(k).cov = covest[k];
        clusterProb.at(k) = (float)nums[k] / (float)sum;

        regularize_cov( clusterParam.at(k).cov, regularization_val);
        ensure_posdef(clusterParam.at(k).cov , regularization_val);
        
        clusterParam.at(k).covdet = 
            compute3x3symm_inverse( 
                    clusterParam.at(k).cov, clusterParam.at(k).covinv);
        // try fixing the situation if somehow ended up neg.def:
        while (clusterParam.at(k).covdet < 0) { 
            regularize_cov( clusterParam.at(k).cov, 1e-1);
            clusterParam.at(k).covdet = 
                compute3x3symm_inverse( 
                        clusterParam.at(k).cov, clusterParam.at(k).covinv);
        }
    }
}

/*! Computes \f$ (p-\mu)^T \Sigma^{-1} (p-\mu) \f$.
    @param p - input LAB value
    @param mu - mean of the GMM component
    @param covinv - precomputed covariance inverse: \f$ \Sigma^{-1}\f$.
    @return value of the expression above.
 */
float GMM::compute_exp_arg(const vector<float>& p, 
        const vector<float>& mu, const vector<float>& covinv) {
    vector<float> q(3);
    float Q;
    q[0] = mu[0]-p[0];
    q[1] = mu[1]-p[1];
    q[2] = mu[2]-p[2];

    // Q = (p-mu)' covinv (p-mu), cov = (a_00, a_01, a_02, a_11, a_12, a_22)
    Q = covinv[0]*q[0]*q[0] + 
        covinv[4]*q[1]*q[1] + 
        covinv[8]*q[2]*q[2] + 
        2*covinv[3]*q[0]*q[1] +
        2*covinv[6]*q[0]*q[2] +
        2*covinv[7]*q[1]*q[2];
    return Q;
}

void GMM::print_means() {
    printf("Means: \n");
    for (int i=0; i<numclusters; ++i) {
        printf("%d) %.3f, %.3f, %.3f\n", i, 
                clusterParam[i].mu[0], 
                clusterParam[i].mu[1],
                clusterParam[i].mu[2] );
    }
}

void GMM::print_mixture_weights() {
    printf("Mixture weights: \n");
    for (int i=0; i<numclusters; ++i)
        printf(" %.3f ", clusterProb[i]);
    printf("\n");
}

void GMM::print_covariances() {
    printf("Covariances: \n");
    for (int i=0; i<numclusters; ++i) {
        for (int u1=0; u1<3; ++u1) {
            for (int u2=0; u2<3; ++u2) {
                printf("%.3f ", clusterParam[i].cov[3*u1+u2]);
            }
            printf("\t");
            for (int u2=0; u2<3; ++u2) {
                printf("%.3f ", clusterParam[i].covinv[3*u1+u2]);
            }
            printf("\n");
        }
        printf("Determinant: %.3f\n\n", clusterParam[i].covdet);
    }
}

vector<float> GMM::init_mu() {
    vector<float> mu(3,0);
    float temp;
    for (int u=0; u<3; ++u) {
        temp = maxval[u]-minval[u];
        mu[u] = minval[u] + 
            temp*((float)(rand() % RAND_MAX))/ ((float)RAND_MAX);
    }
    return mu;
}

vector<float> GMM::init_mu(const vector<float>& img) {
    int n = img.size()/3;
    int k = rand() % n;
    vector<float> mu(3,0);
    memcpy(&mu[0], &img[3*k], sizeof(float)*3 );
    return mu;
}

vector<float> GMM::init_cov() {
    vector<float> cov = vector<float>(9);
    float temp;
    for (int u1=0; u1<3; ++u1) {
        temp = maxval[u1]-minval[u1];
        cov[3*u1+u1] = (float)(temp*temp);
    }
    // set scale automatically if it looks like diagonal elements are zero
    // (for whatever reason)
    if (cov[0]==0) { cov[0] = 1; }
    if (cov[4]==0) { cov[4] = 1; }
    if (cov[8]==0) { cov[8] = 1; }
    return cov;
}

/*! Regularizes matrix by adding a small value along the diagonal. */
void GMM::regularize_cov( vector<float>& m, float val) {
    m[0] += val;
    m[4] += val;
    m[8] += val;
}

/*! Tries to ensure positive-definiteness by first replacing any nans
    (which generally do NOT occur) with zeros, then regularizing by adding
    a value along the diagonal, and then ensuring that off-diagonal entries
    are smaller than the smallest of the diagonal entries:
    \f$ C_{ij} \leq \min(C_{ii},C_{jj}) \f$ (which must hold for covariance
    matrices (Cauchy-Schwarz) but is not sufficient for pos-def-ness.
*/
void GMM::ensure_posdef( vector<float>& m, float val) {
    int nfix = 0;
    // replace nans with zeros.
    for (int u1=0; u1<3; ++u1) {
        for (int u2=0; u2<3; ++u2) {
            if (m[3*u1+u2] != m[3*u1+u2]) {
                m[3*u1+u2] = 0.0;
                nfix++;
            }
        }
    }
    if (nfix>0) {
        regularize_cov( m , val);
    }

    if (abs(m[3]) >= (min(m[0], m[4]))-val) {
        m[1] = m[3] = 
            copysignf(1.0, m[3])*max(0.0f, (min(m[0],m[4]) - val));
    }
    if (abs(m[6]) >= (min(m[0], m[8]))-val) {
        m[2] = m[6] = 
            copysignf(1.0, m[6] )*max(0.0f, (min(m[0],m[8]) - val));
    }
    if (abs(m[7]) >= (min(m[4], m[8]))-val) {
        m[5] = m[7] = 
            copysignf(1.0, m[7] )*max(0.0f, (min(m[4],m[8]) - val));
    }
}

/*! Given an image, estimate and store its range, minimum value, and maximum 
    value for each channel. These values will later be used to initialize means
    and covariances of the gaussian mixture components
*/
void GMM::initImageRange(const vector<float>& img) {
    minval = vector<float>(3, numeric_limits<float>::max() );
    maxval = vector<float>(3, numeric_limits<float>::min() );
    int n = img.size() / 3;
    for (int i=0; i<n; ++i) { 
        for (int k=0; k<3; ++k) {
            minval[k] = min( img[3*i+k], minval[k] );
            maxval[k] = max( img[3*i+k], maxval[k] );
        }
    }
}
