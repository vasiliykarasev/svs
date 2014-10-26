#include <jni.h>
#include <android/log.h>
#include <cstdlib>
#include <vector>
#include <list>

#include "bcv/Slic.h"
#include "bcv/SlicGraph.h"
#include "bcv/bcv_basic.h"
using namespace std;

extern "C" {
#define LOG_TAG "jni_slic"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeRunSlic(
        JNIEnv *env, jclass, jbyteArray data, jint r, jint c, jint chan, jint K, jint M, jint num_iters);
}
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_bcv_svs_Processor_nativeRunSlic(
        JNIEnv *env, jclass, jbyteArray data, jint r, jint c, jint chan, jint K, jint M, jint num_iters) {
    jbyte* c_data = env->GetByteArrayElements(data, 0);
    vector<unsigned char> img( (unsigned char*)c_data, (unsigned char*)c_data + r*c*chan);

    int max_levels = 1;
    float scale = 2;
    Slic slic = Slic(img, r, c, chan, K, M, num_iters, max_levels, scale);
    vector<int> sp = slic.segment(); // basic segmentation call.
    vector<unsigned char> matout = slic.get_boundary_image(img);
    memcpy(c_data, &matout[0], sizeof(unsigned char)*r*c*chan);
    env->ReleaseByteArrayElements(data, c_data, 0);
}
