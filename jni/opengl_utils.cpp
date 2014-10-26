// This file just implements a small utility that writes the camera-frame into
// a ***larger*** (or the same size) texture.
// It is absolutely necessary that the texture is larger.
#include <string.h>
#include <jni.h>
#include <android/log.h>

#ifndef max
#define max(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a < _b ? _a : _b; })
#endif

extern "C"{

#define LOG_TAG "opengl_utils"
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))


JNIEXPORT void JNICALL Java_bcv_svs_GLLayer_clipframe(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jbyteArray poutArray,
		  jint width, jint height, jint width_out, jint height_out);

static inline void clip_common( unsigned char *in_buffer, int in_w, int in_h,
		unsigned char *out_buffer, int out_w, int out_h);
}

JNIEXPORT void JNICALL Java_bcv_svs_GLLayer_clipframe(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jbyteArray poutArray,
		  jint width, jint height, jint out_w, jint out_h) {

        jbyte *inArray;
        jbyte *outArray;
        inArray = env->GetByteArrayElements(pinArray, JNI_FALSE);
        outArray = env->GetByteArrayElements(poutArray, JNI_FALSE);

        // flush with zeros
        // unnecessary, since this memory will never be used
        // (i.e. it is enough to flush it once)
        //memset((unsigned char*)outArray, 0, width*height*3*sizeof(unsigned char) );

        clip_common((unsigned char*)inArray, width, height,
        		(unsigned char*)outArray, out_w, out_h);

        env->ReleaseByteArrayElements(pinArray, inArray, 0);
        env->ReleaseByteArrayElements(poutArray, outArray, 0);
}

static inline void clip_common( unsigned char *in_buffer, int in_w, int in_h,
		unsigned char *out_buffer, int out_w, int out_h) {

	// it is assumed that the output buffer is larger than the input buffer in
	// BOTH dimensions (or at most the same size).
	// moreover, when the output dimensions are strictly larger than the input
	// dimensions, the data is placed in the 'center' of the output array
	// (i.e. there is black padding on the corners).
	// presumably this will make opengl centering \epsilon-less miserable
	int offset_h = (out_h-in_h)/2;
	int offset_w = (out_w-in_w)/2;
	if ((offset_h < 0) || (offset_w < 0)) {
		LOGE("output opengl array smaller than the input frame!");
	}
/*
	for (int i = 0; i < in_h; i++) {
		for (int j = 0; j < in_w*3; j++) {
	    	out_buffer[(i+offset_h)*out_w*3 + offset_w*3 + j] = in_buffer[i*in_w*3 + j];
	    }
	} */
	for (int i = 0; i < in_h; i++) {
		memcpy( out_buffer + (i+offset_h)*out_w*3 + offset_w*3,
				in_buffer  + (i*in_w*3),
				sizeof(unsigned char)*(in_w*3) );
	}
}
