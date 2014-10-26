#include <string.h>
#include <jni.h>
#include <android/log.h>
#ifndef max
#define max(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a < _b ? _a : _b; })
#endif

extern "C"{
#define LOG_TAG "yuv2rgb"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))


JNIEXPORT void JNICALL Java_bcv_svs_Processor_yuv420rgb(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jint width, jint height,
		  jbyteArray ptemp, jint out_w, jint out_h);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_yuv2rgb(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jbyteArray ptemp, jint width, jint height);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_y2gray(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jbyteArray ptemp, jint width, jint height);

JNIEXPORT void JNICALL Java_bcv_svs_Processor_convertaspectratio(JNIEnv* env, jobject object,
		  jbyteArray pintemp, jint width, jint height, jint outwidth, jint outheight);

static inline void color_convert_common(
    unsigned char *pY, unsigned char *pUV, int width, int height,
    unsigned char *buffer, int size, /* buffer size in bytes */
    int gray, int rotate);

static inline void clip_common( unsigned char *in_buffer, int in_w, int in_h,
		unsigned char *out_buffer, int out_w, int out_h);

static inline void convert_y2gray(unsigned char* in, unsigned char* out, int w, int h);

static inline void convert_aspect_ratio(unsigned char* temp,
						int width, int height, int outwidth, int outheight);
}

/*
 * convert a yuv420 array to a rgb array
 */
JNIEXPORT void JNICALL Java_bcv_svs_Processor_yuv420rgb(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jint width, jint height,
		  jbyteArray ptemp, jint out_w, jint out_h) {

        jbyte *inArray;
        jbyte *temp;
        inArray = env->GetByteArrayElements(pinArray, JNI_FALSE);
        temp = env->GetByteArrayElements(ptemp, JNI_FALSE);

        unsigned char* temp_array = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);
        color_convert_common(
        		 (unsigned char*)inArray, (unsigned char*)inArray + width * height,
                 width, height,
                 temp_array, width * height * 3,
                 0, 0);
        clip_common(temp_array, width, height,
        		(unsigned char*)temp,out_w, out_h);

        free(temp_array);
        //release arrays:
        env->ReleaseByteArrayElements(pinArray, inArray, 0);
        env->ReleaseByteArrayElements(ptemp, temp, 0);
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_yuv2rgb(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jbyteArray ptemp, jint width, jint height) {

        jbyte *inArray;
        jbyte *temp;
        inArray = env->GetByteArrayElements(pinArray, JNI_FALSE);
        temp = env->GetByteArrayElements(ptemp, JNI_FALSE);

        color_convert_common(
        		(unsigned char*)inArray, (unsigned char*)inArray + width*height,
                 width, height, (unsigned char*)temp, width*height*3, 0, 0);
        //release arrays:
        env->ReleaseByteArrayElements(pinArray, inArray, 0);
        env->ReleaseByteArrayElements(ptemp, temp, 0);
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_y2gray(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jbyteArray ptemp, jint width, jint height) {

        jbyte *inArray;
        jbyte *temp;
        inArray = env->GetByteArrayElements(pinArray, JNI_FALSE);
        temp = env->GetByteArrayElements(ptemp, JNI_FALSE);

        convert_y2gray((unsigned char*)inArray, (unsigned char*)temp, width, height);
        //release arrays:
        env->ReleaseByteArrayElements(pinArray, inArray, 0);
        env->ReleaseByteArrayElements(ptemp, temp, 0);
}

JNIEXPORT void JNICALL Java_bcv_svs_Processor_convertaspectratio(JNIEnv* env, jobject object,
		  jbyteArray pinArray, jint width, jint height, jint outwidth, jint outheight) {
    jbyte *inArray;
    inArray = env->GetByteArrayElements(pinArray, JNI_FALSE);
    convert_aspect_ratio((unsigned char*)inArray, width, height, outwidth, outheight);
    env->ReleaseByteArrayElements(pinArray, inArray, 0);
}

/*
   YUV 4:2:0 image with a plane of 8 bit Y samples followed by an interleaved
   U/V plane containing 8 bit 2x2 subsampled chroma samples.
   except the interleave order of U and V is reversed.

                        H V
   Y Sample Period      1 1
   U (Cb) Sample Period 2 2
   V (Cr) Sample Period 2 2
 */


/*
 size of a char:
 find . -name limits.h -exec grep CHAR_BIT {} \;
 */
const int bytes_per_pixel = 2;

static inline void color_convert_common( unsigned char *pY, unsigned char *pUV,
    int width, int height, unsigned char *buffer,
    int size, /* buffer size in bytes */
    int gray, int rotate) {
        int i, j;
        int nR, nG, nB;
        int nY, nU, nV;
        unsigned char *out = buffer;
        int offset = 0;
        // YUV 4:2:0
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                nY = *(pY + i * width + j);
                nV = *(pUV + (i/2) * width + bytes_per_pixel * (j/2));
                nU = *(pUV + (i/2) * width + bytes_per_pixel * (j/2) + 1);
                // Yuv Convert
                nY -= 16;
                nU -= 128;
                nV -= 128;

                if (nY < 0)
                    nY = 0;

                nB = (int)(1192 * nY + 2066 * nU);
                nG = (int)(1192 * nY - 833 * nV - 400 * nU);
                nR = (int)(1192 * nY + 1634 * nV);

                nR = min(262143, max(0, nR));
                nG = min(262143, max(0, nG));
                nB = min(262143, max(0, nB));

                nR >>= 10; nR &= 0xff;
                nG >>= 10; nG &= 0xff;
                nB >>= 10; nB &= 0xff;
                out[offset++] = (unsigned char)nR;
                out[offset++] = (unsigned char)nG;
                out[offset++] = (unsigned char)nB;
            }
        }
}
static inline void convert_y2gray(unsigned char* in, unsigned char* out, int w, int h) {
	int n = w*h;
    for (int i = 0; i < n; i++) {
        out[3*i+0] = in[i];
        out[3*i+1] = in[i];
        out[3*i+2] = in[i];
    }
}

static inline void clip_common( unsigned char *in_buffer, int in_w, int in_h,
		unsigned char *out_buffer, int out_w, int out_h) {
	int i, j;

	for (i = 0; i < min(in_h, out_h); i++) {
	     for (j = 0; j < min(in_w, out_w)*3; j++) {
			out_buffer[i*out_w*3 + j] = in_buffer[i*in_w*3 + j];
	     }
	}
}

static inline void convert_aspect_ratio(unsigned char* inArray, int width, int height, int outwidth, int outheight) {

    int height_offset = (height-outheight)/2;
    int width_offset = (width-outwidth)/2;
    if ( (height_offset == 0) && (width_offset==0) ) { return; }

	unsigned char* pY = inArray;
	unsigned char* pUV = inArray + width*height;
	unsigned char* temp = (unsigned char*)malloc(sizeof(unsigned char)*outheight*(outwidth + outwidth) );
    unsigned char* pOutY = temp;
    unsigned char* pOutUV = temp + outwidth*outheight;

	unsigned char nY, nV, nU;
	// YUV 4:2:0
    int i,j,k,l;
    if ((height_offset != 0) && (width_offset == 0)) {
    	for (i = height_offset, k = 0; i < height-height_offset; ++i, ++k) {
			memcpy( pOutY + k*width, pY+i*width, sizeof(unsigned char)*width );
    	}
    	for (i = height_offset, k = 0; i < height-height_offset; i+=2, k+=2) {
			memcpy( pOutUV + k/2*width, pUV + i/2*width, sizeof(unsigned char)*width );
    	}
    }
    else if ((height_offset == 0) && (width_offset != 0)) {
    	for (i = 0; i < height; ++i) {
    		memcpy( pOutY + i*outwidth, pY + i*width + width_offset, sizeof(unsigned char)*outwidth );
    	}
    	for (i = 0; i < height; i+=2) {
    		memcpy( pOutUV + i*outwidth/2, pY + i*width/2 + width_offset/2, sizeof(unsigned char)*outwidth );
		}
    } else if ((height_offset != 0) && (width_offset != 0)) {
		// general annoying case (not optimized).
    	for (i = height_offset, k = 0; i < height-height_offset; ++i, ++k) {
			for (j = width_offset, l = 0; j < width-width_offset; ++j,++l) {
				pOutY[k*outwidth+l] = pY[i * width + j];
				pOutUV[(k/2) * outwidth + bytes_per_pixel * (l/2)+0] = pUV[ (i/2) * width + bytes_per_pixel * (j/2) ];
				pOutUV[(k/2) * outwidth + bytes_per_pixel * (l/2)+1] = pUV[ (i/2) * width + bytes_per_pixel * (j/2) + 1 ];
			}
		}
    }
    memcpy(inArray, temp, sizeof(unsigned char)*(outheight*(outwidth+outwidth)) );
    free(temp);
}
