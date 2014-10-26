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
JNIEXPORT void JNICALL Java_bcv_svs_Processor_imresize(
    JNIEnv *env, jclass, jbyteArray, jint, jint, jbyteArray, jint, jint, jint);

}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_imresize(
    JNIEnv *env, jclass, 
    jbyteArray in, jint in_rows, jint in_cols, 
    jbyteArray out, jint out_rows, jint out_cols, jint chan) {

    jbyte* c_in = env->GetByteArrayElements(in, 0);
    jbyte* c_out = env->GetByteArrayElements(out, 0);


    Mat matin = Mat(in_rows, in_cols, ((chan==3) ? CV_8UC3 : CV_8UC1), c_in);
    Mat matout = Mat(out_rows, out_cols, ((chan==3) ? CV_8UC3 : CV_8UC1), c_out);
    resize(matin, matout, Size(out_cols, out_rows), 0, 0, INTER_NEAREST);

    env->ReleaseByteArrayElements(in, c_in, 0);
    env->ReleaseByteArrayElements(out, c_out, 0);
}

