//! @file bcv_kmeans.h
#ifndef BCV_KMEANS_H_
#define BCV_KMEANS_H_

#include <cstdlib>
#include <assert.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <bitset>
#include <limits>
#include <algorithm>
#include <numeric>

using namespace std;

//! kmeans cluster initialization options
enum {
    KMEANS_INIT_RANDOM=0,
    KMEANS_INIT_FURTHEST_FIRST=1 };

//! kmeans algorithm options
enum {
    KMEANS_LLOYD=0,
    KMEANS_ELKAN=1 };

//! A very basic kmeans implementation.
class bcv_kmeans {
public:
    int num_pts;
    int dim;
    int K;
    int num_iterations; 
    int verbosity;
    float dfx_tolerance; 
    int kmeans_init_method; // initialization type
    int kmeans_method; // lloyd or elkan?
 
    const float* data;
    float* distance; // distance of point to nearest cluster
    int* assignments;
    float* centers;
    int* count; // number of points belonging to cluster

    bcv_kmeans();
    bcv_kmeans(const bcv_kmeans& that);
    bcv_kmeans& operator=(const bcv_kmeans& that);
    bcv_kmeans(const vector<float>& d, int num_pts_, int dim_, int K_, 
            int num_iterations_=100, int verbosity_=0, float dfx_tol=1e-5f, 
            int init_method = KMEANS_INIT_RANDOM, int solve_method = KMEANS_LLOYD);
    bcv_kmeans(const char* fname);
    ~bcv_kmeans(); 
    void init_centers();
    void init_centers_furthest_first();

    void get_centers(vector<float>& data);
    void get_assignments(vector<int>& a);
    void get_assignments(vector<int>& a, const vector<float>&d);

    void kmeans();
    void elkan_kmeans(); 

    void save(const char* fname);
private:
    int check_parameters();
    float eval_function_value();

    float inline get_distance_sq(const float* a, const float* b, int n);

    void elkan_compute_cluster_distance(float* D, float* S);
};

#endif // BCV_KMEANS_H_
