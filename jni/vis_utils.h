#ifndef VIS_UTILS_H_
#define VIS_UTILS_H_

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

void draw_contour_boundary(Mat& img, int* pts_x, int* pts_y, int n, bool finished);

Mat mask_from_contour(int rows, int cols, int* pts, int n);

void draw_edge(uchar* img, const vector<uchar>& mask,
		const vector<uchar>& val, int rows, int cols, int chan);


#endif // VIS_UTILS_H_