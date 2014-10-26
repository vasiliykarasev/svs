#ifndef Slic_H_
#define Slic_H_

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <climits>
#include <vector>
#include <list>
#include <queue>
#include <stack>
#include <set>
#include "bcv_utils.h"

using namespace std;

struct cluster {
    vector<unsigned char> rgb;
    unsigned char gray; // use for grayscale.
    int x;
    int y;
    int valid;
};

//! Simple Linear Iterative Clustering (SLIC) superpixels 
class Slic {
public:
    vector<int> rows;
    vector<int> cols;
    vector<int> search_region_x;
    vector<int> search_region_y;
    int num_levels;   
    float scale_factor;
 
    int chan;
    int K;
    int M;
    int num_iters;
    
    vector<cluster> centers;
    vector<int> d;

    vector<vector<unsigned char> > pyramid;
    vector<vector<int> > assignments_pyramid;

    Slic();
    ~Slic();
    Slic(const vector<unsigned char>& img, int rows_, int cols_, int chan_, 
            int K_=200, int M_=100, int num_iters_=10, int max_levels_=3, float scale_factor_=2);
    void reset_image(const vector<unsigned char>& img);
    vector<int> segment();
    // a few printouts (to std)
    void print_assignment();
    void print_centers();
    // write some data to files
    void write_assignment(const char* fname);
    void write_centers(const char* fname);
    vector <unsigned char> get_boundary_image(const vector<unsigned char>& rgb_data, int level=0);
private:
    int nx, ny;
    vector<int> pixcount; // number of pixels within a superpixel

    void get_superpixel_pixcount(int level);
    void try_fill(int x, int y, int i, vector<bool>& mask, vector<bool>& visited,
        int xlo, int xhi, int ylo, int yhi, int cols_);
    void try_fill_iterative(int i, vector<bool>& mask,
            vector<bool>& visited, int xlo, int xhi, int ylo, int yhi);
    void reassign_neighbors(vector<bool>& mask, 
            int cluster, int xlo, int xhi, int ylo, int yhi, int level);
    void init_centers();
    void adjust_centers();
    void remove_empty_superpixels(int level);
    void ensure_contiguity(int level);
    void assign(int level);
    void update(int level); 

    void imresize(vector<unsigned char>& out, 
            const vector<unsigned char>& img, 
            int rows, int cols, int out_rows, int out_cols);
    void subsample(vector<unsigned char>& out, 
            const vector<unsigned char>& img, int rows, int cols);
    void build_pyramid(const vector<unsigned char>& img, int levels);
    void upsample_cluster_centers(int level);
    void upsample_assignments(int level);

    int get_distance_weight(int level);
};

#endif  // Slic_H_
