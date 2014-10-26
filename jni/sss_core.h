#ifndef SSS_CORE_H
#define SSS_CORE_H

#include "bcv/bcv_utils.h"
#include "bcv/bcv_basic.h"
#include "bcv/tSlic1c.h"
#include "bcv/SlicGraph.h"
#include "bcv/GMM.h"
#include "bcv/bcv_sparse_op.h"
#include <opencv2/core/core.hpp>

//! parameters related to segmentation
struct sss_segment {
	float tv; //! weight on total variation penalty.
    float wt; //! temporal unary weight
    float beta; //! exp(-beta*|I(x)-I(y)|^2 )
	int num_iters;
};

//! parameters related to modeling objects of interest
struct sss_model {
    vector<float> data_fg;
    vector<float> data_bg;
    
    vector<SlicNode<float> > supergraph;
    vector<uchar> supermask;
    int prev_mask_size;

    int num_fg;
    int num_bg;
    int max_num_pts;
};

struct jni_state {
	// -------------------------------------------------------------------------
	//						       superpixellization
	// -------------------------------------------------------------------------
	tslic_params params;
	tSlic tslic;
	cv::Mat previmg; // previous image (for lucas-kanade)
    vector<int> prev_seg_id; // previous superpixel ids
    vector<uchar> prev_seg; // previous segmentation (defined on superpixels)
    vector<uchar> cur_seg; // current segmentation mask (defined on *pixels*)
    // superpixel graph:
    vector<SlicNode<float> > graph;
    vector<int> tslic_ids_prev;
    vector<int> tslic_ids;

    // -------------------------------------------------------------------------
    //                              object model
    // -------------------------------------------------------------------------
    sss_model obj_model;
    // -------------------------------------------------------------------------
    // 								GMMs & segmentation
    // -------------------------------------------------------------------------
    GMM gmm_fg;
    GMM gmm_bg;
    bool gmm_set;
    int gmm_num_clusters;
    int gmm_num_iters;
    // contour data.
    vector<int> contour_pts;
    sss_segment seg;
};

#endif // SSS_CORE_H
