// Example 1: Read an image and perform superpixel segmentation.
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "sss_core.h"

using namespace std;
using namespace cv;

extern "C" {
#define LOG_TAG "jni_tslic"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#define printf LOGI
// this is a temporary function
JNIEXPORT void JNICALL Java_bcv_svs_Processor_drawContourBoundary(
    JNIEnv *env, jclass, jbyteArray, jint, jint, jintArray, jintArray, jint, jboolean);

JNIEXPORT jlong JNICALL Java_bcv_svs_Processor_nativeCreateJniObject(
        JNIEnv *env, jclass, jint slic_K, jint slic_M, 
        jint slic_num_iters, jint seg_K, 
        jfloat seg_tv, jfloat seg_wt, jfloat seg_beta, jint seg_num_iters,
        jint rows, jint cols, jint chan);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeDestroyJniObject(
        JNIEnv *env, jclass, jlong addr);
//! runs slic on the image. resulting segmentation is stored in state->tslic.
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeRunTSlic(
        JNIEnv *env, jclass, jlong addr, jbyteArray data);

//! constructs superpixel graph using image data in 'data'.
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeTSlicConstructGraph(
        JNIEnv *env, jclass, jlong addr, jbyteArray data, int chan);

// visualize slic superpixel boundaries.
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeShowTSlicImage(
        JNIEnv *env, jclass, jlong addr, jbyteArray data, jint chan);
}

JNIEXPORT jlong JNICALL Java_bcv_svs_Processor_nativeCreateJniObject(
        JNIEnv *env, jclass, jint slic_K, jint slic_M, 
        jint slic_num_iters, jint seg_K, 
        jfloat seg_tv, jfloat seg_wt, jfloat seg_beta, jint seg_num_iters,
        jint rows, jint cols, jint chan) {

	jlong addr = (jlong)new jni_state;
	jni_state* state = (jni_state*)addr;
	// slic parameters:
	state->params;
	state->params.rows = rows;
	state->params.cols = cols;
	state->params.chan = chan;
	state->params.K = slic_K;
	state->params.M = slic_M;
	state->params.num_iters = slic_num_iters;
	state->params.min_area_frac = 0.50;
	state->params.max_area_frac = 1.750f;
	state->tslic = tSlic(state->params);
	state->previmg = Mat(0,0,CV_8UC1);
    state->prev_seg = vector<uchar>();
    //state->prev_seg_id = vector<int>();
    state->tslic_ids = vector<int>();
    state->tslic_ids_prev = vector<int>();

    state->graph = vector<SlicNode<float> >();
	// segmentation crap (consider reserving storage here..)
    state->obj_model.num_fg = 0;
    state->obj_model.num_bg = 0;
    state->obj_model.max_num_pts = 3000;
    state->obj_model.data_fg = vector<float>();
    state->obj_model.data_fg.reserve( 3000*3 );
    state->obj_model.data_bg.reserve( 3000*3 );
    state->obj_model.supermask = vector<uchar>();
    state->obj_model.supermask.reserve(3000);
    state->obj_model.supergraph = vector<SlicNode<float> >();
    state->obj_model.supergraph.reserve(3000);
    state->obj_model.prev_mask_size = 0;

    state->contour_pts = vector<int>();
    state->gmm_fg = GMM();
    state->gmm_fg = GMM();
    state->gmm_set = 0;
    // segmentation parameters
    state->gmm_num_clusters = seg_K;
    state->gmm_num_iters = 25;
    state->seg.tv = seg_tv;
    state->seg.wt = seg_wt;
    state->seg.beta = seg_beta;
    state->seg.num_iters = seg_num_iters;

	return addr;
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeDestroyJniObject(
        JNIEnv *env, jclass, jlong addr) {
	delete (jni_state*)addr;
}


JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeRunTSlic(
        JNIEnv *env, jclass, jlong addr, jbyteArray data) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    jbyte* c_data = env->GetByteArrayElements(data, 0);
    // -------------------------------------------------------------------------
    Size LK_winsize = Size(15,15); //Size(21,21);
    int LK_maxpyrlevel = 0;

    int rows = state->params.rows;
    int cols = state->params.cols;
    int chan = state->params.chan;
    int n = rows*cols*chan;
    vector<unsigned char> img( (unsigned char*)c_data, (unsigned char*)c_data + n);
    Mat curmat = Mat(rows, cols, CV_8UC1, c_data);

    if ((rows == 0) || (cols == 0)) { return; }

    if ((state->tslic.K==0) || 
        (state->previmg.rows == 0) || (state->previmg.cols==0))  {
    	// using only the current image
        state->tslic = tSlic(state->params);
    	state->tslic.segment(img, vector<int>() );
	} else {
    	int K = state->tslic.K;
        vector<Point2f> prev_pts = vector<Point2f>( K );
        vector<Point2f> cur_pts = vector<Point2f>( K );
        for (int i = 0; i < K; ++i) {
            prev_pts[i].x = state->tslic.centers[i].x;
            prev_pts[i].y = state->tslic.centers[i].y;
        }

        vector<uchar> status = vector<uchar>( K );
        vector<float> err = vector<float>( K );
        calcOpticalFlowPyrLK(state->previmg, curmat, prev_pts, cur_pts, 
                                    status, err, LK_winsize, LK_maxpyrlevel);

        vector<int> predicted_centers = vector<int>( K*2 );
        for (int i = 0; i < K; ++i) {
            predicted_centers[2*i] = cur_pts[i].x;
            predicted_centers[2*i+1] = cur_pts[i].y;
        }
        state->tslic.segment( img , predicted_centers);
    }
    // store current image:
    curmat.copyTo( state->previmg );

    //int bad = 0;
    //for (int i = 0; i < state->tslic.K; ++i) {
    //    bad += (state->tslic.centers[i].id < 0);
    //}

    // -------------------------------------------------------------------------
    env->ReleaseByteArrayElements(data, c_data, 0);
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeTSlicConstructGraph(
        JNIEnv *env, jclass, jlong addr, jbyteArray data, int chan) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    jbyte* c_data = env->GetByteArrayElements(data, 0);
    // -------------------------------------------------------------------------
    int rows = state->params.rows;
    int cols = state->params.cols;
    int n = rows*cols*chan;
    vector<unsigned char> img( (unsigned char*)c_data, (unsigned char*)c_data + n);
    Mat curmat = Mat(rows, cols, (chan==3)? CV_8UC3 : CV_8UC1, c_data);
    if ((rows == 0) || (cols == 0)) { return; }

    // construct superpixel graph
    vector<int> assignments = state->tslic.get_assignments();
    state->graph =
       construct_slic_graph<float>(img, assignments, rows, cols, chan);
    slic_scale_graph_pixels(state->graph, 1.0f/256.0f );

    state->tslic_ids_prev = state->tslic_ids;
    state->tslic_ids = state->tslic.get_ids();
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeShowTSlicImage(
        JNIEnv *env, jclass, jlong addr, jbyteArray data, jint chan) {
    jni_state* state = (jni_state*)addr;
    if (state == 0) { return; }
    jbyte* c_data = env->GetByteArrayElements(data, 0);
    // -------------------------------------------------------------------------
    int rows = state->params.rows;
    int cols = state->params.cols;
    int n = rows*cols*chan;
    vector<unsigned char> img( (unsigned char*)c_data, (unsigned char*)c_data + n);

    vector<uchar> res = state->tslic.get_boundary_image(img, rows, cols, chan);
    memcpy(c_data, &res[0], sizeof(unsigned char)*n);
    env->ReleaseByteArrayElements(data, c_data, 0);
}
