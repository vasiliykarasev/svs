#ifndef tSlic_H_
#define tSlic_H_

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include <cmath>
#include <cfloat>
#include <climits>
#include <limits>
#include <vector>
#include <list>
#include <queue>
#include <stack>
#include <set>

#include "bcv_utils.h"
#include "bcv_alg.h"

using namespace std;

typedef unsigned long ulong;
typedef unsigned char uchar;

// optional parameter structure
struct tslic_params {
    int K; // nominal number of superpixels
    int M; // geometric penalty
    int num_iters; // number of iterations per-step
    int rows; // image size
    int cols; // image size
    int chan; // image size

    //! lower bound on voxel size (fraction of nominal area). this is in (0,1]
    float min_area_frac; 
    //! upper bound on voxel size (fraction of nominal area). this is in [1, inf)
    float max_area_frac;
};

// superpixel cluster representation
struct tslic_cluster {
    uchar color[3];
    int x;
    int y;
    int id; // unique identifier of the supervoxel in the video
    int lifetime;
};

// node in superpixel graph. this is the output structure
struct tslic_node {
    tslic_cluster data; // mean color, location, id.
    vector<int> neighbors; // index into existing superpixels
    // TODO: do we really need both? is it costly to compute both?
    vector<pair<int,int> > pixels_xy; // row column representation
    vector<int> pixels_linear; // linear index into assigned pixels
};

class tSlic {
public:
    int rows;
    int cols;
    int search_region_x;
    int search_region_y;
 
    int chan;
    int K; // number of superpixels in current frame
    int Knom; // nominal number of superpixels
    int Kmax; // maximum number of superpixels possible.
    int M; // geometric weight (larger-> more compact)
    int num_iters; // number of iterations of kmeans
    int num_voxels_lived; // number of voxels processed so far. 
    
    int min_area; // smallest allowed superpixel (rule for merging)
    int max_area; // largest allowed superpixel (rule for splitting)

    vector<tslic_cluster> centers;
    vector<int> d;

    vector<uchar> img_data;
    vector<int> assignments;

    tSlic();
    ~tSlic();
    tSlic(tslic_params& p);
    
    void segment(const vector<uchar>& img, const vector<int>& pts=vector<int>() );
    
    vector<tslic_node> construct_graph();

    //! returns a vector of superpixel identification numbers
    vector<int> get_ids();
    // a few printouts (to std)
    //void print_assignment();
    //void print_centers();
    // write some data to files
    //void write_assignment(const char* fname);
    //void write_centers(const char* fname);
    //void write_graph(const vector<SlicNode>& graph, const char* fname);
    //void write_graph_geometry(const vector<SlicNode>& graph, const char* fname);
    vector <uchar> get_boundary_image(const vector<uchar>& rgb_data, int rows=0, int cols=0, int chan=0);
    //vector <unsigned char> get_average_image(const vector<SlicNode>& graph);
private:
    int nx, ny;
    vector<int> pixcount; // number of pixels within a superpixel

    void get_superpixel_pixcount();
    void try_fill(int x, int y, int i, vector<bool>& mask, vector<bool>& visited,
        int xlo, int xhi, int ylo, int yhi);
    void reassign_neighbors(vector<bool>& mask, 
            int cluster, int xlo, int xhi, int ylo, int yhi);
    void warp_centers_with_flow(const vector<float>& flow, const vector<uchar>& img);
    void warp_centers_with_tracks(const vector<int>& pts, const vector<uchar>& img);
    
    void init_centers();
    void adjust_centers();
    void remove_empty_superpixels();
    void ensure_contiguity();
    void assign();
    void update(); 
    void increment_lifetime();

    int split_move();
    int merge_move();
    vector<float> compute_superpixel_variance(int k);
    vector<int> compute_superpixel_bounds(int k);

    bool is_cluster_invalid(tslic_cluster& c);
    void make_cluster_invalid(tslic_cluster& c);
    bool is_in_range(int x, int y, int d);    

    void check_parameters(tslic_params& p);
    //int inline linear_index(int r, int c, int k, int cols, int chan);
    //int inline linear_index(int r, int c, int cols);
    //int inline getrow(int i, int cols);
    //int inline getcol(int i, int cols);
};

#endif  // tSlic_H_
