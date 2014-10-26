//! @file bcv_kmeans.cpp
#include "bcv_kmeans.h"

bcv_kmeans::bcv_kmeans() { 
    num_pts = 0;
    dim = 0;
    K = 0;
    num_iterations = 0;
    verbosity = 0;
    dfx_tolerance = 1e-5;
    kmeans_init_method = KMEANS_INIT_RANDOM;
    kmeans_method = KMEANS_LLOYD;

    data = NULL;
    distance = NULL;
    assignments = NULL;
    centers = NULL;
    count = NULL;
}

bcv_kmeans::bcv_kmeans(const bcv_kmeans& that) {
    num_pts = that.num_pts;
    dim = that.dim;
    K = that.K;
    num_iterations = that.num_iterations;
    data = that.data;
    verbosity = that.verbosity;
    dfx_tolerance = that.dfx_tolerance;
    kmeans_init_method = that.kmeans_init_method;
    kmeans_method = that.kmeans_method;

    if (distance) { delete distance; }
    if (assignments) { delete assignments; }
    if (centers) { delete centers; }
    if (count) { delete count; }
    
    distance = new float[num_pts];
    assignments = new int[num_pts];
    centers = new float[K*dim];
    count = new int[K];   
    memcpy(distance, that.distance, sizeof(float)*num_pts);
    memcpy(assignments, that.assignments, sizeof(int)*num_pts);
    memcpy(centers, that.centers, sizeof(float)*dim*K);
    memcpy(count, that.count, sizeof(int)*K);
}

bcv_kmeans& bcv_kmeans::operator=(const bcv_kmeans& that) {
    num_pts = that.num_pts;
    dim = that.dim;
    K = that.K;
    num_iterations = that.num_iterations;
    data = that.data;
    verbosity = that.verbosity;
    dfx_tolerance = that.dfx_tolerance;
    kmeans_init_method = that.kmeans_init_method;
    kmeans_method = that.kmeans_method;

    if (distance) { delete distance; }
    if (assignments) { delete assignments; }
    if (centers) { delete centers; }
    if (count) { delete count; }
    
    distance = new float[num_pts];
    assignments = new int[num_pts];
    centers = new float[K*dim];
    count = new int[K];   
    memcpy(distance, that.distance, sizeof(float)*num_pts);
    memcpy(assignments, that.assignments, sizeof(int)*num_pts);
    memcpy(centers, that.centers, sizeof(float)*dim*K);
    memcpy(count, that.count, sizeof(int)*K);
    return *this;
}

bcv_kmeans::~bcv_kmeans() {
    // data is not ours to delete, so we wont do it.
    if (distance) { delete distance; }
    if (assignments) { delete assignments;}
    if (centers) { delete centers; }
    if (count) { delete count; }
}

//! main constructor, which will also compute kmeans.
//! @param[in] d - input data 
//! @param[in] num_pts - number of data points in 'd'
//! @param[in] dim - dimension of each data point in 'd'
//! @param[in] K - number of clusters
//! @param[in] num_iterations - number of kmeans iterations to run
bcv_kmeans::bcv_kmeans(const vector<float>& d, 
        int num_pts_, int dim_, int K_, int num_iterations_, int verbosity_,
        float dfx_tol, int init_method, int solve_method) {
    // check that parameters are sensible
    num_pts = num_pts_;
    dim = dim_;
    K = K_;
    num_iterations = num_iterations_;
    verbosity = verbosity_;
    dfx_tolerance = dfx_tol;
    kmeans_init_method = init_method;   
    kmeans_method = solve_method; 

    int valid = check_parameters();
    if (valid) { 
        data = d.data(); // should just point to the data provided.
        distance = new float[num_pts];
        assignments = new int[num_pts];
        centers = new float[K*dim];
        count = new int[K];
        if ((distance == 0) || (distance==0) || (assignments==0) ||
            (centers == 0) || (count==0)) {
            printf("malloc failed.\n");
            return;
        }
        // run entire kmeans
        if (kmeans_method == KMEANS_LLOYD) {
            kmeans();
        } else if (kmeans_method == KMEANS_ELKAN) {
            elkan_kmeans();
        } else {
            printf("unrecognized.\n");
        }
    } else {
        data = NULL;
        distance = NULL;
        assignments = NULL;
        centers = NULL;
        count = NULL;
    }
}

//! constructor, which retrieves kmeans parameters and precomputed cluster centers
//! from pre-saved file
bcv_kmeans::bcv_kmeans(const char* fname) {
    FILE* fid = fopen(fname,"r");
    if (fid==NULL) { 
        printf("error reading %s.\n", fname);
        bcv_kmeans();
    }
    int m;   
    m = fread(&num_pts, sizeof(int), 1, fid);
    assert(m == 1);
    m = fread(&dim, sizeof(int), 1, fid);
    assert(m == 1);
    m = fread(&K, sizeof(int), 1, fid);
    assert(m == 1);
    m = fread(&num_iterations, sizeof(int), 1, fid);
    assert(m == 1);
    m = fread(&verbosity, sizeof(int), 1, fid);
    assert(m == 1);
    dfx_tolerance = 1e-5;
 
    if (!check_parameters()) { 
        fclose(fid);
        bcv_kmeans();
    }

    // initialize all arrays.
    data = NULL;
    distance = new float[num_pts];
    assignments = new int[num_pts];
    centers = new float[K*dim];
    count = new int[K];
        
    m = fread(centers, sizeof(float), K*dim, fid);
    assert(m==K*dim);
    fclose(fid);
}

int bcv_kmeans::check_parameters() {
    int valid = 1;
    if ((num_pts < 0) || (K < 0) || (dim < 0) || (num_iterations < 0)) { 
        printf("Arguments to bcv_kmeans are not sensible.\n");
        num_pts = 0;
        K = 0;
        dim = 0;
        valid = 0;
    }
    if (num_pts < K) { 
        printf("Arguments to bcv_kmeans imply # clusters > # pts?\n");
        valid = 0;
    } 
    return valid;
}

void bcv_kmeans::init_centers() { 
    // choose K out of N indices at random.
    vector<int> init_idx;
    init_idx.reserve(K);
    // TODO: this is really awkward; need to change to set.
    vector<bool> bs = vector<bool>(num_pts, 0);;
    int sofar = 0;
    while (sofar < K) { 
        int id = rand() % num_pts;
        if (!bs[id]) {
            bs[id]=1;
            init_idx.push_back(id);
            sofar++;
        }
    }
    // initialize centers using random data points.
    for (int k = 0; k < K; ++k) { 
        for (int d = 0; d < dim; ++d) { 
            centers[d + dim*k] = data[d+dim*init_idx[k]];
        }
    }
}

//! Furthest first initialization described in Dasgupta 2002.
//! \f$ C^{k+1} = \arg\min_x \min_{j = 1,...,k} d(x,C^j) \f$
void bcv_kmeans::init_centers_furthest_first() {
    // initialize first cluster to be the mean of the data.
    for (int d = 0; d < dim; ++d) { centers[d] = 0; }
    for (int i = 0; i < num_pts; ++i) { 
        for (int d = 0; d < dim; ++d) { centers[d] += data[d+dim*i]; }
    }
    for (int d = 0; d < dim; ++d) { centers[d] /= float(num_pts); }
    
    // heuristic described in Elkan 2003, and earlier in Dasgupta 2002. 
    vector<float> mind = vector<float>(num_pts, 0.0f);
    for (int i = 0; i < num_pts; ++i) {
        mind[i] = get_distance_sq(centers, data+i*dim, dim);
    }
    for (int k = 1; k < K; ++k) { 
        // find index of maximum element in the mind vector.
        // this will be: argmax_x min_c d(x,c)
        int idx = std::distance(mind.begin(), min_element(mind.begin(), mind.end() ));
        // set the cluster to that data point:
        for (int d = 0; d < dim; ++d) { centers[d+k*dim] = data[d+idx*dim]; }
        // update vector of min. distances:
        for (int i = 0; i < num_pts; ++i) {
            float dist = get_distance_sq( centers+k*dim, data+i*dim, dim);
            mind[i] = min(mind[i], dist);
        }
    }
}

//! Runs the Lloyd variant of k-means
void bcv_kmeans::kmeans() { 
    float fx = numeric_limits<float>::max();
    float fx_prev = fx;
    float dfx = 0;

    // initialize clusters
    if (kmeans_init_method == KMEANS_INIT_RANDOM) {
        init_centers();
    } else if (kmeans_init_method == KMEANS_INIT_FURTHEST_FIRST) {
        init_centers_furthest_first();
    }
    for (int t = 0; t < num_iterations; ++t) { 
        //-- assign stage
        for (int i = 0; i < num_pts; ++i) {
            distance[i] = numeric_limits<float>::max();
        }
        for (int k = 0; k < K; ++k) { 
            for (int i = 0; i < num_pts; ++i) { 
                float d = get_distance_sq( data+i*dim, centers+k*dim, dim);
                if (d < distance[i]) {
                    distance[i] = d;
                    assignments[i] = k;
                }
            }
        }
        //-- update stage
        memset(count, 0, sizeof(int)*K);
        for (int i = 0; i < num_pts; ++i) {
            count[ assignments[i] ]++;
        }
        // fix empty clusters:
        for (int k = 0; k < K; ++k) { 
            if (count[k]==0) {
                // choose a point at random, assign it to this cluster
                int id = -1;
                int c = k;
                while (count[c]<=1) { // (avoid creating another empty cluster)
                    id = rand() % num_pts;
                    c = assignments[id];
                }
                count[c]--;
                count[k] = 1;
                assignments[id] = k;
            }
        }
        memset(centers, 0.0f, sizeof(float)*dim*K );
        for (int i = 0; i < num_pts; ++i) { 
            int k = assignments[i];
            for (int u = 0; u < dim; ++u) { 
                centers[u + dim*k] += data[u + dim*i];
            }
        }
        for (int k = 0; k < K; ++k) { 
            for (int u = 0; u < dim; ++u) { 
                centers[u + dim*k]/=float(count[k]);
            }
        }

        // track performance via function value
        fx_prev = fx;
        fx = eval_function_value();
        dfx = abs(fx-fx_prev) / (fx+1e-8f);
        if (verbosity > 0) { 
            printf("bcv_kmeans: %03d : f(x) = %7.5g \t df(x) = %7.5g\n",
                    t, fx, dfx);
        }
        if (dfx < dfx_tolerance) {  
            if (verbosity > 0) { 
                printf("bcv_kmeans: reached tolerance level; stopping.\n");
            }
            break;
        }
    }
}

//! Updates the distances between clusters (step in Elkan variant)
//! @param D - matrix of distances among clusters
//! @param S - distance to the nearest cluster, for each cluster
void bcv_kmeans::elkan_compute_cluster_distance(float* D, float* S) {
    for (int k1 = 0; k1 < K; ++k1) { 
        D[k1 + K*k1] = 0.0f;
        for (int k2 = 0; k2 < k1; ++k2) { 
            float dist = get_distance_sq(centers+k1*dim, centers+k2*dim, dim);
            D[k2 + K*k1] = D[k1 + K*k2] = sqrt(dist);
        }
    }
    for (int k1 = 0; k1 < K; ++k1) {
        S[k1] = numeric_limits<float>::max();
        for (int k2 = 0; k2 < K; ++k2) {
            if (k2 == k1) { continue; }
            S[k1] = min(S[k1], D[k1 + k2*K]);
        }
        S[k1] *= 0.5f;
    }
}

//! Runs the Elkan variant of k-means
void bcv_kmeans::elkan_kmeans() {
    // init several variables specific to elkan-variant
    float fx = numeric_limits<float>::max();
    float fx_prev = fx;
    float dfx = 0;
    // initialize clusters
    if (kmeans_init_method == KMEANS_INIT_RANDOM) {
        init_centers();
    } else if (kmeans_init_method == KMEANS_INIT_FURTHEST_FIRST) {
        init_centers_furthest_first();
    } else {
        printf("not supported; using random method.\n");
        init_centers();
    }
    // ------------------------------------------------------------------------
    //              fast initial point assignments to clusters:
    // ------------------------------------------------------------------------
    float* lowerbound = new float[num_pts*K];
    float* upperbound = new float[num_pts];
    bool* R = new bool[num_pts];
    float* S = new float[K];
    float* Dcenters = new float[K*K];
    float* temp_centers = new float[dim*K];
    float* dist_shift = new float[K];
    if ((lowerbound==0) || (upperbound ==0) || (R==0) || (S==0) || 
        (Dcenters==0) || (temp_centers==0) || (dist_shift==0)) {
        printf("malloc in kmeans failed.\n");
        if (lowerbound !=0) { delete lowerbound; }
        if (upperbound !=0) { delete upperbound; }
        if (R != 0) { delete R; }
        if (S != 0) { delete S; }
        if (Dcenters != 0) { delete Dcenters; }
        if (temp_centers != 0) { delete temp_centers; }
        if (dist_shift != 0) { delete dist_shift; }
        return;
        // TODO:
        // actually here should exit the program altogether.
        // since any subsequent result of kmeans will be invalid.
    }
    
    

    elkan_compute_cluster_distance(Dcenters, S);
    for (int i = 0; i < num_pts; ++i) { 
        int cur_k = -1;
        float mind = numeric_limits<float>::max();
        for (int k = 0; k < K; ++k) { 
            // fast distance check:
            if ((cur_k != -1) && (Dcenters[k+cur_k*K] > 2*mind)) {
                continue;
            }
            // compute distance:
            float d = get_distance_sq(centers+k*dim, data+i*dim, dim);
            d = sqrt(d);
            lowerbound[i + k*num_pts] = d; 
            if (d < mind) { 
                mind = d;
                cur_k = k;
            }
        }
        upperbound[i] = mind; // min_c d(x,c)
        assignments[i] = cur_k;
    }
    memset(R, 1, sizeof(bool)*num_pts);
    for (int t = 0; t < num_iterations; ++t) {
        // --------------------------------------------------------------------
        //          fast update of cluster assignments
        // --------------------------------------------------------------------
        elkan_compute_cluster_distance(Dcenters, S);
        // based on the bound (lemma 1 in the paper, decide which points
        // will not change cluster assignments
        int nskipped = 0;
        for (int i = 0; i < num_pts; ++i) {
            if (upperbound[i] < S[ assignments[i] ]) { nskipped++; continue; }
            int id = assignments[i];
            float d_to_cluster = 0.0f;
            if (R[i]) {
                R[i] = 0;
                d_to_cluster = get_distance_sq(centers+id*dim, data+i*dim, dim);
            } else {
                d_to_cluster = upperbound[i]*upperbound[i];
            }
            for (int k = 0; k < K; ++k) { 
                float rhs = min( lowerbound[i+k*num_pts], 0.5f*Dcenters[id+K*k] );
                if (d_to_cluster < rhs*rhs) {
                    continue;
                }
                float dxc = get_distance_sq(centers+k*dim, data+i*dim, dim);
                if (dxc < d_to_cluster) { // both distances are squared
                    assignments[i] = k;
                    d_to_cluster = dxc;
                }
            }       
        }
        // --------------------------------------------------------------------
        //
        // --------------------------------------------------------------------
        memset(count, 0, K*sizeof(int) );
        for (int i = 0; i < num_pts; ++i) {
            count[ assignments[i] ]++;
        }
        // fix empty clusters:
        for (int k = 0; k < K; ++k) { 
            if (count[k]==0) {
                // choose a point at random, assign it to this cluster
                int id = -1;
                int c = k;
                while (count[c]<=1) { // (avoid creating another empty cluster)
                    id = rand() % num_pts;
                    c = assignments[id];
                }
                count[c]--;
                count[k] = 1;
                assignments[id] = k;
            }
        }
        memset(temp_centers, 0.0f, dim*K*sizeof(float) );
        for (int i = 0; i < num_pts; ++i) { 
            int k = assignments[i];
            for (int u = 0; u < dim; ++u) { 
                temp_centers[u + dim*k] += data[u + dim*i];
            }
        }
        for (int k = 0; k < K; ++k) { 
            for (int u = 0; u < dim; ++u) { 
                temp_centers[u + dim*k]/=float(count[k]);
            }
        }
        //---------------------------------------------------------------------
        //                       update lower bound 
        //---------------------------------------------------------------------
        for (int k = 0; k < K; ++k) {
            float d = get_distance_sq(temp_centers+k*dim, centers+k*dim, dim);
            d = sqrt(d);
            for (int i = 0; i < num_pts; ++i) { 
                float lb = lowerbound[i+k*num_pts] - d;
                lowerbound[i+k*num_pts] = max(lb, 0.0f); 
            }
        } 
        //---------------------------------------------------------------------
        //                        update upper bound 
        //---------------------------------------------------------------------
        for (int k = 0; k < K; ++k) { 
            float d = get_distance_sq(temp_centers+k*dim, centers+k*dim, dim);
            dist_shift[k] = sqrt(d);
        }
        for (int i = 0; i < num_pts; ++i) {
            int id = assignments[i];
            upperbound[i] += dist_shift[id];
            R[i] = (dist_shift[id]>0);
        }
        memcpy( centers, temp_centers, sizeof(float)*K*dim);
        //---------------------------------------------------------------------
        //              track performance via function value
        //---------------------------------------------------------------------
        fx_prev = fx;
        fx = eval_function_value();
        dfx = abs(fx-fx_prev) / (fx+1e-8f);
        if (verbosity > 0) { 
            printf("bcv_kmeans: %03d : f(x) = %7.5g \t df(x) = %7.5g nskip = %d\n", 
                    t, fx, dfx, nskipped);
        }
        if (dfx < dfx_tolerance) {  
            if (verbosity > 0) { 
                printf("bcv_kmeans: reached tolerance level; stopping.\n");
            }
            break;
        }
    }

    delete lowerbound;
    delete upperbound;
    delete R;
    delete S;
    delete Dcenters;
    delete temp_centers;
    delete dist_shift;
}

float inline bcv_kmeans::get_distance_sq(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i) { 
        d+= (a[i]-b[i])*(a[i]-b[i]);
    }
    return d;
}

//! Returns cluster centers, as a vector with size K*dim.
//! @param[out] c - output cluster centers 
void bcv_kmeans::get_centers(vector<float>& c) { 
    c = vector<float>(K*dim);
    memcpy(c.data(), centers, sizeof(float)*K*dim);
}

//! Returns cluster assignments of each point.
//! @param[out] a - output cluster assignments
void bcv_kmeans::get_assignments(vector<int>& a) { 
    a = vector<int>(num_pts);
    memcpy(a.data(), assignments, sizeof(int)*num_pts);
}

//! Returns cluster assignments for each point in d.
//! @param[out] a - output cluster assignments
//! @param[in] d - input data
void bcv_kmeans::get_assignments(vector<int>& a, const vector<float>& data) { 
    int num_pts = data.size() / dim; // number of points.
    a = vector<int>(num_pts);
    // for each point, find nearest cluster
    for (int i = 0; i < num_pts; ++i) { 
        float mind = numeric_limits<float>::max();
        for (int k = 0; k < K; ++k) { 
            float d = get_distance_sq(&data[i*dim], centers+k*dim, dim);
            if (d < mind) {
                a[i] = k;
                mind = d;
            }
        }
    }
}

//! Evaluates \f$ \sum_{k=1}^K \sum_{i \in C(k) }\| x_i - \mu_k \|^2 \f$
float bcv_kmeans::eval_function_value() {
    float val = 0.0f;
    for (int i = 0; i < num_pts; ++i) { 
        int k = assignments[i];
        float d = get_distance_sq(data+i*dim, centers+k*dim, dim);
        val += (d / float(dim));
    }
    return val;
}


//! Writes kmeans parameters and cluster centers to file.
void bcv_kmeans::save(const char* fname) {
    FILE* fid = fopen(fname,"w");
    if (fid==NULL) { 
        printf("error writing to %s.\n", fname);
        return;
    }
    fwrite(&num_pts, sizeof(int), 1, fid);
    fwrite(&dim, sizeof(int), 1, fid);
    fwrite(&K, sizeof(int), 1, fid);
    fwrite(&num_iterations, sizeof(int), 1, fid);
    fwrite(&verbosity, sizeof(int), 1, fid);
    // dont write data; we dont own it.
    // dont write 'distance', 'assignments', or 'count' - intermediate variables
    // do write 'centers', these are needed to push points to nearest
    fwrite(centers, sizeof(float), K*dim, fid);

    fclose(fid);
}
