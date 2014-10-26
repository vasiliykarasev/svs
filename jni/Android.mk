LOCAL_PATH := $(call my-dir)
# -----------------------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE    := yuv2rgb
LOCAL_SRC_FILES := yuv2rgb.cpp
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_ARM_NEON := true
include $(BUILD_SHARED_LIBRARY)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE    := opengl_utils
LOCAL_SRC_FILES := opengl_utils.cpp
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_ARM_NEON := true
include $(BUILD_SHARED_LIBRARY)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE    := opencv_tools
LOCAL_SRC_FILES := opencv/src/core/algorithm.cpp
LOCAL_SRC_FILES += opencv/src/core/alloc.cpp
LOCAL_SRC_FILES += opencv/src/core/arithm.cpp
LOCAL_SRC_FILES += opencv/src/core/array.cpp
LOCAL_SRC_FILES += opencv/src/core/cmdparser.cpp
LOCAL_SRC_FILES += opencv/src/core/convert.cpp
LOCAL_SRC_FILES += opencv/src/core/copy.cpp
LOCAL_SRC_FILES += opencv/src/core/datastructs.cpp
LOCAL_SRC_FILES += opencv/src/core/drawing.cpp
LOCAL_SRC_FILES += opencv/src/core/dxt.cpp
LOCAL_SRC_FILES += opencv/src/core/glob.cpp
LOCAL_SRC_FILES += opencv/src/core/gpumat.cpp
LOCAL_SRC_FILES += opencv/src/core/lapack.cpp
LOCAL_SRC_FILES += opencv/src/core/mathfuncs.cpp
LOCAL_SRC_FILES += opencv/src/core/matmul.cpp
LOCAL_SRC_FILES += opencv/src/core/matop.cpp
LOCAL_SRC_FILES += opencv/src/core/matrix.cpp
LOCAL_SRC_FILES += opencv/src/core/opengl_interop.cpp
LOCAL_SRC_FILES += opencv/src/core/opengl_interop_deprecated.cpp
LOCAL_SRC_FILES += opencv/src/core/out.cpp
LOCAL_SRC_FILES += opencv/src/core/parallel.cpp
LOCAL_SRC_FILES += opencv/src/core/persistence.cpp
LOCAL_SRC_FILES += opencv/src/core/rand.cpp
LOCAL_SRC_FILES += opencv/src/core/stat.cpp
LOCAL_SRC_FILES += opencv/src/core/system.cpp
LOCAL_SRC_FILES += opencv/src/core/tables.cpp
#
LOCAL_SRC_FILES += opencv/src/highgui/precomp.hpp
LOCAL_SRC_FILES += opencv/src/highgui/utils.cpp
#
LOCAL_SRC_FILES += opencv/src/imgproc/accum.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/approx.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/canny.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/clahe.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/color.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/contours.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/convhull.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/corner.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/cornersubpix.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/deriv.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/distransform.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/featureselect.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/filter.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/geometry.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/imgwarp.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/linefit.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/matchcontours.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/moments.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/morph.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/phasecorr.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/pyramids.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/rotcalipers.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/samplers.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/shapedescr.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/smooth.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/subdivision2d.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/sumpixels.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/tables.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/templmatch.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/thresh.cpp
LOCAL_SRC_FILES += opencv/src/imgproc/utils.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/emd.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/floodfill.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/gabor.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/generalized_hough.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/grabcut.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/histogram.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/hough.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/undistort.cpp
#LOCAL_SRC_FILES += opencv/src/imgproc/segmentation.cpp
#
LOCAL_SRC_FILES += opencv/src/video/lkpyramid.cpp

LOCAL_LDLIBS +=  -llog -ldl
LOCAL_CPPFLAGS := -std=c++0x
LOCAL_EXPORT_C_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/opencv
LOCAL_EXPORT_CPP_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/opencv
LOCAL_C_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/opencv
LOCAL_CPP_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/opencv
LOCAL_ARM_NEON := true
include $(BUILD_SHARED_LIBRARY)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 									basic opencv utilities
# -----------------------------------------------------------------------------
#include $(CLEAR_VARS)
#export MAINDIR:= $(LOCAL_PATH)
#LOCAL_MODULE := jni_opencv_utils
#LOCAL_SRC_FILES := jni_opencv_utils.cpp
#LOCAL_LDLIBS +=  -llog -ldl
#LOCAL_CPPFLAGS := -std=c++0x
#LOCAL_SHARED_LIBRARIES += opencv_tools
#LOCAL_ARM_NEON := true
#include $(BUILD_SHARED_LIBRARY)
#
# -----------------------------------------------------------------------------
# 									basic tslic
# -----------------------------------------------------------------------------
include $(CLEAR_VARS)
export MAINDIR:= $(LOCAL_PATH)

LOCAL_MODULE := jni_tslic
LOCAL_SRC_FILES := jni_tslic.cpp bcv/bcv_basic.cpp bcv/bcv_utils.cpp bcv/tSlic1c.cpp bcv/GMM.cpp bcv/bcv_alg.cpp
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_CPPFLAGS := -std=c++0x
LOCAL_SHARED_LIBRARIES += opencv_tools
LOCAL_EXPORT_C_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/bcv/
LOCAL_EXPORT_CPP_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/bcv/
LOCAL_ARM_NEON := true
include $(BUILD_SHARED_LIBRARY)

# -----------------------------------------------------------------------------
# 								basic segmentation routines
# -----------------------------------------------------------------------------
include $(CLEAR_VARS)
export MAINDIR:= $(LOCAL_PATH)

LOCAL_MODULE := jni_segmentation
LOCAL_SRC_FILES := jni_segmentation.cpp vis_utils.cpp segmentation_utils.cpp 
LOCAL_SRC_FILES += bcv/bcv_basic.cpp bcv/bcv_utils.cpp bcv/GMM.cpp
LOCAL_SRC_FILES += bcv/bcv_sparse_op.cpp bcv/tvsegmentbinary.cpp bcv/bcv_kmeans.cpp
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_CPPFLAGS := -std=c++0x
LOCAL_SHARED_LIBRARIES += opencv_tools
LOCAL_EXPORT_C_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/bcv/
LOCAL_EXPORT_CPP_INCLUDES += /home/vasiliy/Sandbox/workspace/svs/jni/bcv/
LOCAL_ARM_NEON := true
include $(BUILD_SHARED_LIBRARY)