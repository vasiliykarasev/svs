// jni_segmentation.cpp
#include <jni.h>
#include <android/log.h>
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

#include "segmentation_utils.h"
#include "vis_utils.h"

using namespace std;
using namespace cv;

extern "C" {
#define LOG_TAG "jni_segmentation"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#define printf LOGI


JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeLearnAddFeatures(
            JNIEnv *env, jclass, jlong addr, jint x, jint y, jint w, jint h);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeLearnGMM(
        JNIEnv *env, jclass, jlong addr, jint chan);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeLearnGMMfromData(
        JNIEnv *env, jclass, jlong addr);

JNIEXPORT bool JNICALL Java_bcv_svs_Processor_nativeGetGMMdata(
        JNIEnv *env, jclass, jlong addr, 
        jfloatArray mu_fg, jfloatArray mu_bg, 
        jfloatArray pi_fg, jfloatArray pi_bg);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_doBinarySegmentation(
        JNIEnv *env, jclass, jlong);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_showSegmentationResult(
        JNIEnv *env, jclass, jlong addr, jbyteArray rgbimg, jint rows, jint cols, jint chan);

JNIEXPORT int JNICALL Java_bcv_svs_Processor_nativeGetNumPointsFG(
        JNIEnv *env, jclass, jlong);

JNIEXPORT int JNICALL Java_bcv_svs_Processor_nativeGetNumPointsBG(
        JNIEnv *env, jclass, jlong);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeFinalizeLearning(
        JNIEnv *env, jclass, jlong);

// this is a temporary function
JNIEXPORT void JNICALL Java_bcv_svs_Processor_drawContourBoundary(
    JNIEnv *env, jclass, jbyteArray data, jint rows, jint cols, jint chan,
    jintArray x, jintArray y, jint n, jboolean finished);


JNIEXPORT void JNICALL Java_bcv_svs_Processor_drawGraphEdges(
    JNIEnv *env, jclass, jlong, jbyteArray, jint);

// Writes contour data (xy points) to 'state' structure. 
// The subsequent call to runTSLIC is responsible for processing this data, 
// generating a mask, and ultimately performing the learning, using this data.
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeSetContourData(
        JNIEnv *env, jclass, jlong addr, jintArray x, jintArray y, jint n);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_showSegmentationUnary(
        JNIEnv *env, jclass, jlong, jbyteArray, jint);
}



//! grows superpixel graph / supermask, while the user is holding down the
//! 'learn object' button.
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeLearnAddFeatures(
            JNIEnv *env, jclass, jlong addr, jint x, jint y, jint w, jint h) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    if (state->graph.size() == 0) { return; }

    int x_lo = x - 0.5*w;
    int x_hi = x + 0.5*w;
    int y_lo = y - 0.5*h;
    int y_hi = y + 0.5*h;

    vector<uchar> mask = vector<uchar>( state->graph.size() );

    // compute mask from the rectangle shown on the screen.
    for (size_t i = 0; i < state->graph.size(); ++i) {
        int x = state->graph[i].x;
        int y = state->graph[i].y;
        mask[i] = ((x > x_lo) && (x < x_hi) && (y > y_lo) && (y < y_hi));
    }

    // append to the currently existing mask.
    vector<int> p1;
    vector<int> p2;
    //spgraph_get_id_pairs(p1, p2, state->tslic_ids_prev, state->tslic_ids);
    transform(p1.begin(), p1.end(), p1.begin(), 
            bind1st(plus<int>(), state->obj_model.supergraph.size() - state->obj_model.prev_mask_size) );
    transform(p2.begin(), p2.end(), p2.begin(), 
            bind1st(plus<int>(), state->obj_model.supergraph.size()) );

    // append current mask and current graph.
    state->obj_model.supermask.insert( state->obj_model.supermask.end(),
                                                mask.begin(), mask.end() );
    spgraph_append( state->obj_model.supergraph, state->graph );

    // append temporal neighbors:
    spgraph_add_neighbors( state->obj_model.supergraph, p1, p2);
    state->obj_model.prev_mask_size = mask.size();
}


/*
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeLearnGMM(
        JNIEnv *env, jclass, jlong addr, jint chan) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    if (state->contour_pts.size() == 0) {
        return; // no data to learn GMM from....
    }
    if (state->graph.size()==0) { return; }
    int rows = state->params.rows;
    int cols = state->params.cols;

    int numpts = state->contour_pts.size()/2;
    Mat matmask = mask_from_contour(rows, cols,
                                &state->contour_pts[0], numpts);
    state->contour_pts.clear();
    //
    vector<uchar> mask( (uchar*)matmask.data, (uchar*)matmask.data + rows*cols);
    vector<uchar> mask_vec =
                spgraph_img2vec<uchar, float>(state->graph, mask, rows, cols, 1);

    int m = accumulate(mask_vec.begin(), mask_vec.end(), 0);
    if (m > (state->gmm_num_clusters+1)) {
        update_bgfg_data(state->obj_model, mask_vec, state->graph);

        if (0) {
			learn_appearance_gmm(state->gmm_fg, state->gmm_bg,
				state->gmm_num_clusters, state->gmm_num_iters, state->obj_model);
        } else {
			learn_appearance_gmm_kmeans(state->gmm_fg, state->gmm_bg,
				state->gmm_num_clusters, state->gmm_num_iters, state->obj_model);
		}
    }
} */

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeLearnGMMfromData(
        JNIEnv *env, jclass, jlong addr) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    int m = state->gmm_num_clusters+1;
    if (state->obj_model.data_fg.size() < m) { return; }
    if (state->obj_model.data_bg.size() < m) { return; }

    int rows = state->params.rows;
    int cols = state->params.cols;

    if (0) {
        learn_appearance_gmm(state->gmm_fg, state->gmm_bg,
            state->gmm_num_clusters, state->gmm_num_iters, state->obj_model);
    } else {
        learn_appearance_gmm_kmeans(state->gmm_fg, state->gmm_bg,
            state->gmm_num_clusters, state->gmm_num_iters, state->obj_model);
    }
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeFinalizeLearning(
        JNIEnv *env, jclass, jlong addr) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    if (state->obj_model.supergraph.size() == 0) { return; }
    if (state->obj_model.supermask.size() == 0) { return; }

    // perform iterative GMM estimation. Once finished, the GMMs are estimated,
    // and the model structure is cleared.
    int num_reestimates = 1;
    tvseg_iterative_gmm_estimation(state->gmm_fg, state->gmm_bg, 
            state->obj_model, state->gmm_num_clusters, state->gmm_num_iters, 
            num_reestimates, state->seg.num_iters, state->seg.beta, state->seg.tv);
    
    state->obj_model.data_fg.clear();
    state->obj_model.data_bg.clear();
    state->obj_model.supergraph.clear();
    state->obj_model.supermask.clear();
    state->obj_model.prev_mask_size = 0;
    state->obj_model.num_fg = 0;
    state->obj_model.num_bg = 0;                
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_doBinarySegmentation(
        JNIEnv *env, jclass, jlong addr) {
    double t1;
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    if (state->gmm_fg.clusterProb.size() == 0) { return; }
    if (state->gmm_bg.clusterProb.size() == 0) { return; }
    LOGI("doBinarySegmentation");
    // -------------------------------------------------------------------------
    // create problem structure for solving:
    tvsegmentbinary_params p;
    p.D = create_difference_op(state->graph);
    
    //t1 = now_ms();
    p.unary = compute_unary_potential(state->graph, state->gmm_fg, state->gmm_bg);
    //LOGI("unary: %f ms", now_ms()-t1);

    // compute temporal weight.
    if ((state->seg.wt > 0) && (state->prev_seg.size()>0) && 
                            (state->tslic_ids_prev.size()>0)) {
        vector<float> t_unary = compute_temporal_unary_potential(state->tslic.centers,
            state->prev_seg, state->tslic_ids_prev);
        
        transform(t_unary.begin(), t_unary.end(), t_unary.begin(),
                        bind1st(multiplies<float>(), state->seg.wt ));

        transform(p.unary.begin(), p.unary.end(), t_unary.begin(),
                               p.unary.begin(), plus<float>() );
    }
    //t1 = now_ms();
    p.weights = compute_pairwise_potential(state->graph, state->seg.beta, p.D.nrows);
    // factor-in the TV penalty weight.
    transform(p.weights.begin(), p.weights.end(), p.weights.begin(),
                   bind1st(multiplies<float>(), state->seg.tv) );
    //LOGI("pairwise: %f ms", now_ms()-t1);

    p.nnodes = p.unary.size();
    p.nedges = p.weights.size();
    p.max_iters = state->seg.num_iters;

    tvsegmentbinary tvs = tvsegmentbinary(&p);
    //t1 = now_ms();
    vector<uchar> res_vec = tvs.get_segmentation();
    //LOGI("segmentation: %f ms", now_ms()-t1);
    
    // -------------------------------------------------------------------------
    //      store current segmentation (for use with temporal penalty)
    // -------------------------------------------------------------------------
    if (state->seg.wt > 0) { state->prev_seg = res_vec; }
    // -------------------------------------------------------------------------

    //t1 = now_ms();
    int rows = state->params.rows;
    int cols = state->params.cols;    
    state->cur_seg = spgraph_vec2img( state->graph, res_vec, rows, cols );
}

// rows cols are unused now.
JNIEXPORT void JNICALL Java_bcv_svs_Processor_showSegmentationResult(
        JNIEnv *env, jclass, jlong addr, jbyteArray rgbimg, jint rows, jint cols, jint chan) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    int in_rows = state->params.rows;
    int in_cols = state->params.cols;
    if (state->cur_seg.size() != in_rows*in_cols) { return; }

    vector<uchar> val = vector<uchar>(chan);
    if (chan == 1) { val[0] = 255; }
    if (chan == 3) { val[0] = 86; val[1] = 180; val[2] = 211; }

    jbyte* c_rgbimg = env->GetByteArrayElements(rgbimg, 0);
    draw_edge((uchar*)c_rgbimg, state->cur_seg, val, in_rows, in_cols, chan);

    env->ReleaseByteArrayElements(rgbimg, c_rgbimg, 0);
}


JNIEXPORT bool JNICALL Java_bcv_svs_Processor_nativeGetGMMdata(
        JNIEnv *env, jclass, jlong addr,
        jfloatArray mu_fg, jfloatArray mu_bg,
        jfloatArray pi_fg, jfloatArray pi_bg) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return 0; }

    jfloat* c_mu_fg = env->GetFloatArrayElements(mu_fg, 0);
    jfloat* c_mu_bg = env->GetFloatArrayElements(mu_bg, 0);
    jfloat* c_pi_fg = env->GetFloatArrayElements(pi_fg, 0);
    jfloat* c_pi_bg = env->GetFloatArrayElements(pi_bg, 0);

    int num_clusters = state->gmm_fg.clusterProb.size(); // this should be stored in state..
    int u = 0;
    for (int k = 0; k < num_clusters; ++k) {
        size_t m = state->gmm_fg.clusterParam[k].mu.size();
        for (size_t i = 0; i < m; ++i) {
            c_mu_fg[u] = state->gmm_fg.clusterParam[k].mu[i];
            c_mu_bg[u] = state->gmm_bg.clusterParam[k].mu[i];
            u++;
        }
    }
    for (int k = 0; k < num_clusters; ++k) {
        c_pi_fg[k] = state->gmm_fg.clusterProb[k];
        c_pi_bg[k] = state->gmm_bg.clusterProb[k];
    }

    env->ReleaseFloatArrayElements(mu_fg, c_mu_fg, 0);
    env->ReleaseFloatArrayElements(mu_bg, c_mu_bg, 0);
    env->ReleaseFloatArrayElements(pi_fg, c_pi_fg, 0);
    env->ReleaseFloatArrayElements(pi_bg, c_pi_bg, 0);
    return (num_clusters > 0);
}


JNIEXPORT int JNICALL Java_bcv_svs_Processor_nativeGetNumPointsFG(
        JNIEnv *env, jclass, jlong addr) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return 0; }
    return state->obj_model.num_fg;
}
JNIEXPORT int JNICALL Java_bcv_svs_Processor_nativeGetNumPointsBG(
        JNIEnv *env, jclass, jlong addr) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return 0; }
    return state->obj_model.num_bg;
}


JNIEXPORT void JNICALL Java_bcv_svs_Processor_drawGraphEdges(
    JNIEnv *env, jclass, jlong addr, jbyteArray data, jint chan) {
    jni_state* state = (jni_state*)addr;
    LOGI("drawGraphEdges");

    if (state == 0) { return; }
    jbyte* c_data = env->GetByteArrayElements(data, 0);

    int rows = state->params.rows;
    int cols = state->params.cols;
    vector<uchar> img( (uchar*)c_data, 
                       (uchar*)c_data + rows*cols*chan);

    int nedges = spgraph_get_num_edges(state->graph);
    vector<float> weights = compute_pairwise_potential(state->graph, state->seg.beta, nedges);

    spgraph_vis_edges(img, state->graph, weights, rows, cols, chan);
    memcpy(c_data, &img[0], sizeof(uchar)*rows*cols*chan);

    env->ReleaseByteArrayElements(data, c_data, 0);
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_drawContourBoundary(
    JNIEnv *env, jclass, jbyteArray data, jint rows, jint cols, jint chan,
    jintArray x, jintArray y, jint n, jboolean finished) {

    jbyte* c_data = env->GetByteArrayElements(data, 0);
    jint* x_data = env->GetIntArrayElements(x, 0);
    jint* y_data = env->GetIntArrayElements(y, 0);

    Mat curmat = Mat(rows, cols, ((chan==3) ? CV_8UC3 : CV_8UC1), c_data);

    draw_contour_boundary(curmat, x_data, y_data, n, finished);

    env->ReleaseIntArrayElements(y, y_data, 0);
    env->ReleaseIntArrayElements(x, x_data, 0);
    env->ReleaseByteArrayElements(data, c_data, 0);
}


JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeSetContourData(
        JNIEnv *env, jclass, jlong addr, jintArray x, jintArray y, jint n) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    jint* x_data = env->GetIntArrayElements(x, 0);
    jint* y_data = env->GetIntArrayElements(y, 0);

    if (state->contour_pts.size() > 0) {
        state->contour_pts.clear();
    }
    state->contour_pts = vector<int>(2*n);
    for (int i = 0; i < n; ++i) {
        state->contour_pts[2*i] = x_data[i];
        state->contour_pts[2*i+1] = y_data[i];
    }

    env->ReleaseIntArrayElements(y, y_data, 0);
    env->ReleaseIntArrayElements(x, x_data, 0);
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_showSegmentationUnary(
        JNIEnv *env, jclass, jlong addr, jbyteArray rgbimg, jint chan) {
    double t1;
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    if (state->gmm_fg.clusterProb.size() == 0) { return; }
    if (state->gmm_bg.clusterProb.size() == 0) { return; }
    LOGI("showSegmentationUnary");

    jbyte* c_rgbimg = env->GetByteArrayElements(rgbimg, 0);
    // -------------------------------------------------------------------------
    int rows = state->params.rows;
    int cols = state->params.cols;

    vector<float> unary = compute_unary_potential(state->graph, state->gmm_fg, state->gmm_bg);
    // convert to 255 levels.
    float maxval = *max_element(unary.begin(), unary.end());
    float minval = *min_element(unary.begin(), unary.end());
    
    float norm = max(abs(maxval), abs(minval) );
    vector<uchar> qunary = vector<uchar>( unary.size() );
    for (size_t i = 0; i < unary.size(); ++i) {
        //qunary[i] = saturate_cast<uchar>( 255.0f*(unary[i]-minval)/(maxval-minval) );
        qunary[i] = saturate_cast<uchar>( 255.0f*(unary[i]-norm)/(2*norm) );
    }
    vector<uchar> qunaryimg = spgraph_vec2img( state->graph, qunary, rows, cols );
    if (chan == 1) {
        memcpy(c_rgbimg, &qunaryimg[0], rows*cols*sizeof(uchar) );
    } else if (chan==3) {
        Mat I0 = Mat(rows, cols, CV_8UC1, &qunaryimg[0]);
        cvtColor(I0, I0, CV_GRAY2BGR);
        memcpy(c_rgbimg, I0.data, rows*cols*3*sizeof(uchar));
    } else {
        LOGE("what");
    }
    env->ReleaseByteArrayElements(rgbimg, c_rgbimg, 0);
}
