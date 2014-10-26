#ifndef SEGMENTATION_UTILS_H_
#define SEGMENTATION_UTILS_H_

#include <cstdlib>
#include <vector>
#include <list>

#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "sss_core.h"
#include "bcv/bcv_kmeans.h"
#include "bcv/tvsegmentbinary.h"

#include <android/log.h>
extern "C" {
#define LOG_TAG "segmentation_utils"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
}

using namespace std;
using namespace cv;

bcv_sparse_op<int> create_difference_op(vector<SlicNode<float> >& graph);

//! Evaluates log( p_bg(x)/p_fg(x) )
vector<float> compute_unary_potential(const vector<SlicNode<float> >& graph, 
															GMM& fg, GMM& bg);

//! Returns a vector of pairwise weights: w(x,y) = exp(-beta*|I(x)-I(y)|^2 )
vector<float> compute_pairwise_potential(const vector<SlicNode<float> >& graph, 
                                                    float beta, int nedges);

vector<float> compute_temporal_unary_potential(const vector<tslic_cluster>& nodes,
        const vector<uchar>& old_seg, const vector<int>& old_id);


void learn_appearance_gmm(GMM& fg, GMM& bg, int K, int num_iters, 
														const sss_model& obj);

void learn_appearance_gmm_kmeans(GMM& fg, GMM& bg, int K, int num_iters, 
														const sss_model& obj);

void set_gmm_parameters(GMM& gmm, const vector<float>& data, 
											vector<int>& assignments, int K);

void update_bgfg_data(sss_model& obj, 
		    const vector<uchar>& mask_vec, const vector<SlicNode<float> >& g);

void tvseg_iterative_gmm_estimation(GMM& gmm_fg, GMM& gmm_bg,
				sss_model& obj_model, int gmm_num_clusters, int gmm_num_iters, 
				int num_reestimate_iters, int num_segment_iters,
				float beta, float lambda);

#endif // SEGMENTATION_UTILS_H_
