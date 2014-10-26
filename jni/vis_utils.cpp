#include "vis_utils.h"

void draw_contour_boundary(Mat& img, int* pts_x, int* pts_y, int n, bool finished) { 

	Scalar colour = Scalar::all(128);
    if (img.channels()==3) { colour = Scalar(86, 180, 211); }

	for (int k = 0; k < (n-1); ++k) {
        Point pt1 = Point( pts_x[k], pts_y[k] );
        Point pt2 = Point( pts_x[k+1], pts_y[k+1] );

        line(img, pt1, pt2, colour, 1);
    }
    if ((n > 1) && (finished)) { 
        Point pt1 = Point( pts_x[n-1], pts_y[n-1] );
        Point pt2 = Point( pts_x[0], pts_y[0] );
        line(img, pt1, pt2, colour, 1);
    }

    int* pts = new int[2*n];
    for (int i = 0; i < n; ++i) {
    	pts[2*i] = pts_x[i];
    	pts[2*i+1] = pts_y[i];
    }
    if (finished) {
    	Mat mask = mask_from_contour(img.rows, img.cols, pts, n);
    	mask = mask * 255;
    	if (img.channels()==3) { cvtColor(mask, mask, CV_GRAY2RGB); }
    	img = img + mask;
    }
}

Mat mask_from_contour(int rows, int cols, int* pts, int n) { 
    Mat mask = Mat(rows, cols, CV_8UC1);
    mask = mask*0;
    vector<Point> vpts = vector<Point>(n);
    for (int k = 0; k < n; ++k) { vpts[k] = Point(pts[2*k], pts[2*k+1]); }
    fillConvexPoly(mask, vpts, n, 1);
    double maxVal = 0;     
    minMaxLoc( mask, NULL, &maxVal, NULL, NULL);
    maxVal = max(1.0, maxVal); // avoid division by zero
    mask = mask / (uchar)maxVal;

    return mask;
}

void draw_edge(uchar* img, const vector<uchar>& mask,
					const vector<uchar>& val, int rows, int cols, int chan) {
	char ADD_VAL = 80;
    for (int i = 0; i < mask.size(); ++i) {
		int r = getrow(i, cols);
		int c = getcol(i, cols);

		int ir = linear_index(r, c+1, cols);
		int il = linear_index(r, c-1, cols);
		int it = linear_index(r-1, c, cols);
		int ib = linear_index(r+1, c, cols);

		int case1 = (c<cols-1) && (mask[i]!=mask[ir]);
		int case2 = (c>0) && (mask[i]!=mask[il]);
		int case3 = (r<rows-1) && (mask[i]!=mask[ib]);
		int case4 = (r>0) && (mask[i]!=mask[it]);

		if (case1 || case2 || case3 || case4) {
			for (int k = 0; k < chan; ++k) {
				img[ linear_index(r,c,k,cols,chan) ] = val[k];
			}
		} else if (mask[i]) {
			for (int k = 0; k < chan; ++k) {
				int u = linear_index(r,c,k,cols,chan);
				img[u] = min(255, img[u]+ADD_VAL);
			}
		}
	}
}
