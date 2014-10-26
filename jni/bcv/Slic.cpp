//! @file Slic.cpp
#include "Slic.h"

Slic::Slic() {
}

Slic::~Slic() {
}

Slic::Slic(const vector<unsigned char>& img, int rows_, int cols_, int chan_, 
        int K_, int M_, int num_iters_, int max_levels, float scale_factor_) {
    #ifdef SLIC_DEBUG
    printf("Slic::constructor\n");
    #endif
    // infer *actual* number of superpixels.
    float ar = float(cols_)/float(rows_);
    nx = sqrt( ar*float(K_) );
    ny = sqrt( float(K_)/ar );
    K = nx*ny;
    // figure out the size of each superpixel:
    float step_x = float(cols_) / float(nx+1.0);
    float step_y = float(rows_) / float(ny+1.0);
    if (scale_factor_ <= 1.0) {
        printf("Invalid scale factor (should be >1). Setting to 2.0\n");
        scale_factor_ = 2.0;
    }
    scale_factor = scale_factor_;
    //-------------------------------------------------------------------------
    // figure out how many levels are needed.
    float mindim = (float)min(step_x, step_y);
    int k = 1;
    while (mindim > 4) {
        mindim /= scale_factor;
        k++; 
    }
    if (k < max_levels) {
        printf("Only allowing %d levels\n", k);
    }
    num_levels = min(k, max_levels);
    rows = vector<int>(num_levels);
    cols = vector<int>(num_levels);
    search_region_x = vector<int>(num_levels);
    search_region_y = vector<int>(num_levels);
    
    rows[0] = rows_;
    cols[0] = cols_;
    search_region_x[0] = round(step_x*2.0);
    search_region_y[0] = round(step_y*2.0);
    for (int i = 1; i < num_levels; ++i) { 
        rows[i] = rows[i-1]/scale_factor;
        cols[i] = cols[i-1]/scale_factor;
        search_region_x[i] = search_region_x[i-1]/scale_factor;
        search_region_y[i] = search_region_y[i-1]/scale_factor;
    }
    chan = chan_; // marshall

    int n = rows_*cols_;
    d = vector<int>(n, INT_MAX); // this can be initialized at the lowest level.
    // will have a vector for this..
    
    M = M_ * (chan); // change M with number of channels..
    num_iters = num_iters_;
    // initialize data

    // create pyramids for images and assignment vectors
    build_pyramid(img, num_levels); 
}

void Slic::reset_image(const vector<unsigned char>& img) {
    // warning! if the new array has different dimensions (rows/cols/chans)..
    // ..well there are no checks for that.
    build_pyramid(img, num_levels);
}

vector<int> Slic::segment() {
    init_centers();
    adjust_centers(); 

    assign(num_levels-1);
    for (int level = (num_levels-1); level >= 0; level--) {
        for (int i=0; i<num_iters; ++i) {
            update(level);
            assign(level);
        }
        ensure_contiguity(level);
        if (level > 0) { 
            upsample_cluster_centers(level);
            upsample_assignments(level);
        } else {
            //remove_empty_superpixels(level);
        }
    }
    // copy and return assignments:
    vector<int> *assignments = &assignments_pyramid[0];
    vector<int> out = (*assignments);
    return out;
}

void Slic::init_centers() {
    // initialized at the smallest/coarsest level.
    #ifdef SLIC_DEBUG
    printf("Slic::init_centers()\n");
    #endif
    centers = vector<cluster>(K);

    float step_x = float(cols[num_levels-1]) / float(nx+1.0);
    float step_y = float(rows[num_levels-1]) / float(ny+1.0);
    float offset_x = step_x / 2.0;
    float offset_y = step_y / 2.0;
    int cols_ = cols[num_levels-1];
    vector<unsigned char>* data = &pyramid[num_levels-1];
 
    int i=0;
    if (chan == 3) { 
        for (int x=0; x<nx; ++x) {
            for (int y=0; y<ny; ++y) { 
                centers[i].x = (x)*step_x+offset_x;
                centers[i].y = (y)*step_y+offset_y;
                centers[i].rgb = vector<unsigned char>(3,0);
                for (int u=0; u<3; ++u) {
                    int k = linear_index(y, x, u, cols_, 3);
                    centers[i].rgb[u] = (*data)[k];  
                }
                i++;
            }
        }
    } else {
        for (int x=0; x<nx; ++x) {
            for (int y=0; y<ny; ++y) { 
                centers[i].x = x*step_x + offset_x;
                centers[i].y = y*step_y + offset_y;
                int k = linear_index(y, x, cols_);
                centers[i].gray = (*data)[k];
                i++;
            }
        }
    }
    // initialize to 'valid'
    for (int k = 0; i < K; ++i) {
        centers[k].valid = 1;
    }
}

void Slic::adjust_centers() {
#ifdef SLIC_DEBUG
    printf("Slic::adjust_centers ");
    unsigned long t1,t2;
    t1 = now_us();
#endif
    vector<unsigned char>* data = &pyramid[num_levels-1];
    int cols_ = cols[num_levels-1];
    //int rows_ = rows[num_levels-1];
    for (int k=0; k < K; ++k) { 
        int x = centers[k].x;
        int y = centers[k].y;
        int i=0;
        int min_val = INT_MAX;
        int best_x=x;
        int best_y=y;

        // boundary checks are not implemented because centers should be
        // initialized away from the image boundary in the first place!!           
        if (chan == 3) {
            // for rgb images
            for (int yy=y-1; yy<=(y+1); yy++) {
                for (int xx=x-1; xx<=(x+1); xx++) {
                    float grad = 0.0;
                    for (int u=0; u<3; ++u) {
                        int i_bot = linear_index(yy+1, xx, u, cols_, 3);
                        int i_top = linear_index(yy-1, xx, u, cols_, 3);
                        int i_lef = linear_index(yy, xx-1, u, cols_, 3);
                        int i_rig = linear_index(yy, xx+1, u, cols_, 3);
                        float v = (int)((*data)[i_bot]-(int)(*data)[i_top]);
                        float h = (int)((*data)[i_lef]-(int)(*data)[i_rig]);
                        grad += v*v + h*h;
                    }
                    if (grad < min_val) {
                        min_val = grad;
                        best_x = xx;
                        best_y = yy;
                    }
                    i++;
                }
            }
            centers[k].x = best_x;
            centers[k].y = best_y;
        } else {
            // for grayscale images
            for (int yy=y-1; yy<=(y+1); yy++) {
                for (int xx=x-1; xx<=(x+1); xx++) {
                    float grad = 0.0;
                    int i_bot = linear_index(yy+1, xx, cols_);
                    int i_top = linear_index(yy-1, xx, cols_);
                    int i_lef = linear_index(yy, xx-1, cols_);
                    int i_rig = linear_index(yy, xx+1, cols_);
                    float v = (int)((*data)[i_bot]-(int)(*data)[i_top]);
                    float h = (int)((*data)[i_lef]-(int)(*data)[i_rig]);
                    grad += v*v + h*h;
                    if (grad < min_val) {
                        min_val = grad;
                        best_x = xx;
                        best_y = yy;
                    }
                    i++;
                }
            }
            centers[k].x = best_x;
            centers[k].y = best_y;
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}

int Slic::get_distance_weight(int level) {
    float Mp_temp = float(M);
    for (int u = 0; u < level; ++u) {
        Mp_temp *= (scale_factor*scale_factor);
    }
    return round(Mp_temp);
}

void Slic::assign(int level) {
#ifdef SLIC_DEBUG
    printf("Slic::assign ");
    unsigned long t1,t2;
    t1 = now_us();
#endif
    int rows_ = rows[level];
    int cols_ = cols[level];
    int search_region_x_ = search_region_x[level];
    int search_region_y_ = search_region_y[level];
    int n = rows_*cols_;
    vector<unsigned char> *data = &pyramid[level];
    vector<int> *assignments = &assignments_pyramid[level];
    int Mp = get_distance_weight(level);
   
    for (int i = 0; i < n; ++i) { 
        (*assignments)[i] = 0;
    } 
    for (int i=0; i<n; ++i)
        d[i] = INT_MAX; 
    for (int k=0; k<K; ++k) {
        int xc = centers[k].x;
        int yc = centers[k].y;   

        int xlo = max(0, xc-search_region_x_ );
        int xhi = min(cols_-1, xc+search_region_x_ );
        int ylo = max(0, yc-search_region_y_ );
        int yhi = min(rows_-1, yc+search_region_y_ );

        if (chan == 3) { 
            int r = centers[k].rgb[0];
            int g = centers[k].rgb[1];
            int b = centers[k].rgb[2]; 
            for (int y=ylo; y<=yhi; ++y) {
                int ddy = Mp*(y-yc)*(y-yc);
                for (int x=xlo; x<=xhi; ++x) {
                    int i = linear_index(y, x, cols_);
                    if (ddy >= d[i])
                        continue;
                    int dist = Mp*(x-xc)*(x-xc) + ddy;                
                    if (dist >= d[i])
                        continue;                
                    int q = linear_index(y, x, 0, cols_, 3);
                    int temp1 = ((int)(*data)[q+0]-r);
                    int temp2 = ((int)(*data)[q+1]-g);
                    int temp3 = ((int)(*data)[q+2]-b);
                    dist += (temp1*temp1 + temp2*temp2 + temp3*temp3);

                    if (dist < d[i]) {
                        d[i] = dist;
                        (*assignments)[i] = k;  
                    }
                }
            }
        } else {
            int g = centers[k].gray;
            for (int y=ylo; y<=yhi; ++y) {
                int ddy = Mp*(y-yc)*(y-yc);
                for (int x=xlo; x<=xhi; ++x) {
                    int i = linear_index(y, x, cols_);
                    if (ddy >= d[i])
                        continue;
                    int dist = Mp*(x-xc)*(x-xc) + ddy;                
                    if (dist >= d[i])
                        continue;
                    int temp = ((int)(*data)[i]-g); 
                    dist += temp*temp; 

                    if (dist < d[i]) {
                        d[i] = dist;
                        (*assignments)[i] = k;  
                    }
                }
            }
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}

// SEEMS OK.
void Slic::update(int level) {
#ifdef SLIC_DEBUG
    printf("Slic::update ");
    unsigned long t1,t2; 
    t1 = now_us();
#endif
    int rows_ = rows[level];
    int cols_ = cols[level];
    int search_region_x_ = search_region_x[level];
    int search_region_y_ = search_region_y[level];
    //int n = rows_*cols_;
    vector<unsigned char> *data = &pyramid[level];
    vector<int> *assignments = &assignments_pyramid[level];

    for (int k=0; k<K; ++k) {
        int xc = centers[k].x;
        int yc = centers[k].y;   

        int xlo = max(0, xc-search_region_x_);
        int xhi = min(cols_-1, xc+search_region_x_);
        int ylo = max(0, yc-search_region_y_);
        int yhi = min(rows_-1, yc+search_region_y_);

        if (chan == 3) { 
            int r=0;
            int g=0;    
            int b=0;
            int px=0;
            int py=0;
            int nums=0;
            for (int y=ylo; y<=yhi; ++y) {
                for (int x=xlo; x<=xhi; ++x) {
                    int i = linear_index(y, x, cols_);
                    if ( (*assignments)[i]==k) { 
                        int q = linear_index(y, x, 0, cols_, 3);
                        r += (int)(*data)[q+0];
                        g += (int)(*data)[q+1];
                        b += (int)(*data)[q+2];
                        px += x;
                        py += y;
                        nums++;
                    }
                }
            }

            if (nums>0) {
                centers[k].x = px / nums; 
                centers[k].y = py / nums;
                centers[k].rgb[0] = (unsigned char)(r / nums);
                centers[k].rgb[1] = (unsigned char)(g / nums);
                centers[k].rgb[2] = (unsigned char)(b / nums);
                centers[k].valid = 1;
            } else {
                centers[k].valid = 0;
            }
        } else {
            int r=0;
            int px=0;
            int py=0;
            int nums=0;
            for (int y=ylo; y<=yhi; ++y) {
                for (int x=xlo; x<=xhi; ++x) {
                    int i = linear_index(y, x, cols_);
                    if ( (*assignments)[i]==k) { 
                        nums++;
                        r += (int)(*data)[i];
                        px += x;
                        py += y;
                    }
                }
            }
            if (nums>0) {
                centers[k].x = px / nums; 
                centers[k].y = py / nums;
                centers[k].gray = (unsigned char)(r / nums);
                centers[k].valid = 1;
            } else {
                centers[k].valid = 0;
            }
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}

void Slic::ensure_contiguity(int level) {
#ifdef SLIC_DEBUG
    unsigned long t1, t2;
    t1 = now_us();
    printf("Slic::ensure_contiguity ");
#endif
    int rows_ = rows[level];
    int cols_ = cols[level];
    int search_region_x_ = search_region_x[level];
    int search_region_y_ = search_region_y[level];
    int n = rows_*cols_;
    vector<int> *assignments = &assignments_pyramid[level];

    vector<bool> mask = vector<bool>(n);
    vector<bool> visited = vector<bool>(n);
    for (int k=0; k<K; ++k) {
        int xc = centers[k].x;
        int yc = centers[k].y;
        if ((xc < 0) || (yc < 0)) {
            continue;
        }

        int xlo = max(0, xc-search_region_x_);
        int xhi = min(cols_-1, xc+search_region_x_);
        int ylo = max(0, yc-search_region_y_);
        int yhi = min(rows_-1, yc+search_region_y_);
        // the actual extents of the superpixel.
        int xlosp = xc;
        int xhisp = xc;
        int ylosp = yc;
        int yhisp = yc;
        // by definition, no need to check outside search region.
        int nums = 0;
        int origin_x = -1;
        int origin_y = -1;
        int dist_to_origin = INT_MAX;
        // first check if the centroid (or its 4 nearest neighbors belong to 
        // this superpixel. if yes, great -- we will start at this location.
        // otherwise we need to find a point that belongs to this superpixel
        // while being close to the center (this would be necessary if the 
        // superpixel is U-shaped)
        int i = linear_index(yc, xc, cols_);
        if ( (*assignments)[i]==k ) { 
            // yay!
            origin_x = xc;
            origin_y = yc;
            nums = 1;
        } else {
            for (int y=ylo; y<=yhi; ++y) {
                int ddy = (y-yc)*(y-yc);
                if (ddy > dist_to_origin) { continue; }
                for (int x=xlo; x<=xhi; ++x) {
                    int i = linear_index(y, x, cols_);
                    if ((*assignments)[i]==k) { 
                        int d = ddy + (x-xc)*(x-xc);
                        if (d < dist_to_origin) {
                            dist_to_origin = d;
                            origin_x = x;
                            origin_y = y;
                        }
                    }
                }
            } 
        }

        for (int y = ylo; y<=yhi; ++y) { 
            for (int x=xlo; x<=xhi; ++x) {
                int i = linear_index(y, x, cols_);
                visited[i] = false;
                mask[i] = ( (*assignments)[i] == k);
                if (mask[i]) {
                    nums++;
                    xlosp = min(xlosp, x);
                    ylosp = min(ylosp, y);
                    xhisp = max(xhisp, x);
                    yhisp = max(yhisp, y);
                }
            }
        }
        if (nums==0) {
            //printf("No pixels associated with this superpixel");
            continue;
            // this is a can of worms, but it is handled later
        }
        // we know the point closest to center that belongs to this cluster,
        // so start here and try to 'unset' all the 'set' pixels in the mask.
        try_fill(origin_x, origin_y, linear_index(origin_y, origin_x, cols_),
                mask, visited, xlosp, xhisp, ylosp, yhisp, cols_);
        
        int num_left = 0;
        for (int y=ylosp; y<=yhisp; ++y) {
            for (int x=xlosp; x<=xhisp; ++x) {
                int i = linear_index(y, x, cols_);
                if (mask[i]) {
                    num_left += 1;
                    break;
                }
            }
            if (num_left > 0) { break; }
        }
        
        if (num_left > 0) {
            // assign the remaining pixels to neighboring clusters
            reassign_neighbors(mask, k, xlosp, xhisp, ylosp, yhisp, level);
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}

void Slic::reassign_neighbors(vector<bool>& mask, 
        int cluster, int xlo, int xhi, int ylo, int yhi, int level) {
    int cols_ = cols[level];
    int rows_ = rows[level];
    vector<int> *assignments = &assignments_pyramid[level];
    vector<int> stack;
    stack.reserve(1000);
    // need to reassign pixels.
    for (int y=ylo; y<=yhi; ++y) {
        for (int x=xlo; x<=xhi; ++x) {
            int i = linear_index(y, x, cols_);
            if (!mask[i]) {
                continue;
            } else {
                mask[i] = false;
            }                    
            
            int neighbor=-1;

            int i_rig = linear_index(y, x+1, cols_);
            int i_top = linear_index(y-1, x, cols_);
            int i_bot = linear_index(y+1, x, cols_);
            int i_lef = linear_index(y, x-1, cols_);

            if ( ((*assignments)[i_lef] != cluster) && (x>0) )
                neighbor = (*assignments)[i_lef];
            else if ( ((*assignments)[i_rig] != cluster) && (x<cols_) )
                neighbor = (*assignments)[i_rig];
            else if ( ((*assignments)[i_top] != cluster) && (y>0) )
                neighbor = (*assignments)[i_top];
            else if ( ((*assignments)[i_bot] != cluster) && (y<rows_) )
                neighbor = (*assignments)[i_bot];
            if (neighbor >= 0) {
                // empty the stack...
                (*assignments)[i] = neighbor;
                while (!stack.empty()) {
                    int s = stack[0];
                    stack.erase( stack.begin() );
                    (*assignments)[s] = neighbor;
                }
            } else {
                // push pixel onto the stack...
                stack.push_back( i );
            }
        }
    }
}

void Slic::try_fill(int x, int y, int i,
        vector<bool>& mask, vector<bool>& visited,
        int xlo, int xhi, int ylo, int yhi, int cols_) {

    visited[i] = true;
    bool mask_val = mask[i];
    mask[i] = false;
    if (mask_val) {
        // go left
        int i_lef = linear_index(y, x-1, cols_);
        if ((x > xlo) && (visited[i_lef] == false))
            try_fill(x-1, y, i_lef, mask, visited, xlo, xhi, ylo, yhi, cols_);
        // go right
        int i_rig = linear_index(y, x+1, cols_);
        if ((x < xhi) && (visited[i_rig] == false))
            try_fill(x+1, y, i_rig, mask, visited, xlo, xhi, ylo, yhi, cols_);
        // go bottom
        int i_bot = linear_index(y+1, x, cols_);
        if ((y < yhi) && (visited[i_bot] == false))
            try_fill(x, y+1, i_bot, mask, visited, xlo, xhi, ylo, yhi, cols_);
        // go top
        int i_top = linear_index(y-1, x, cols_);
        if ((y > ylo) && (visited[i_top] == false))
            try_fill(x, y-1, i_top, mask, visited, xlo, xhi, ylo, yhi, cols_);
    }
}

//! TODO URGENT: introduce get superpixel counts into this function.
//! this will allow filtering truly empty superpixels.
void Slic::remove_empty_superpixels(int level) {
    // clean-up routine.
    // if a superpixel is emtpy, her (x,y,lab) value is nan
    // (due to division by zero in update() fn).
    int cols_ = cols[level];
    int rows_ = rows[level];
    int n = rows_*cols_;
    vector<int> *assignments = &assignments_pyramid[level];
     
    get_superpixel_pixcount(level);
    vector<bool> valid(K, true);
    vector<int> mapping(K);
    int numlost = 0;
    for (int i = 0; i < K; ++i) {
        int invalid = (pixcount[i]==0);
        /*
        int invalid = ((centers[i].x < 0) && (centers[i].y < 0));
        if (chan==3) {
            invalid = (
            (centers[i].rgb[0] == 0) &&
            (centers[i].rgb[1] == 0) &&
            (centers[i].rgb[2] == 0) );
        } else {
            invalid = centers[i].gray == 0;
        }*/
        if (invalid) {
            valid[i] = false;
            numlost++;
            mapping[i] = -1;
        } else {
            valid[i] = true;
            mapping[i] = i - numlost;
        }
    }
    // lucky case.
    if (numlost == 0) {
        return;
    }
    // fix superpixel assignments
    // the reason this sometimes crashes apparently is the following:
    // it could be (somehow) that the center is reported as negative, despite
    // the fact that the superpixel contains elements (i.e. not entirely empty).
    // this 'guess' needs to be verified....
    for (int i = 0; i < n; ++i) {
        int newidx = mapping[ (*assignments)[i] ];
        if (newidx == -1) { 
            printf("Major error in Slic::remove_empty_superpixels");
        }
        (*assignments)[i] = newidx;
    }
    // fix superpixel clusters:
    int i_new = 0;
    int i_old = 0;
    for (int u = 0; u < K; ++u) {
        if (valid[i_old]) {
            if (chan == 3) { 
                centers[i_new].rgb[0] = centers[i_old].rgb[0];
                centers[i_new].rgb[1] = centers[i_old].rgb[1];
                centers[i_new].rgb[2] = centers[i_old].rgb[2];
            } else {
                centers[i_new].gray = centers[i_old].gray;
                centers[i_new].x = centers[i_old].x;
                centers[i_new].y = centers[i_old].y;
            }    
            i_new++; // both are incremented
            i_old++;
        } else {
            i_old ++;
        }
    }
    // in principle, 'd' should be fixed analogously. however we never use it
    // so it is not done.
    centers.erase( centers.end() - numlost, centers.end() );
    K = K - numlost;
}

/* The most naive implementation of warping (bilinear) */
void Slic::imresize(vector<unsigned char>& out, 
        const vector<unsigned char>& img, int rows, int cols, int out_rows, int out_cols) {
    // this is needed to prevent floor/ceil returning the same value if xx is integer.
    float EPS = 1e-4; 
    float xx, yy;  // location and flow directions
    float weight1, weight2, weight3, weight4;
    int nx0, nx1, ny0, ny1;  // neighboring location

    float scale = (float(out_rows)/float(rows)+float(out_cols)/float(cols))/2.0;
    if (out.size() != (size_t)out_rows*out_cols*chan) {
        out = vector<unsigned char>(out_rows*out_cols*chan);
    }

    float antiscale = 1.0/scale;
    for (size_t r = 0; r < (size_t)out_rows; ++r) {
        for (size_t c = 0; c < (size_t)out_cols; ++c) {
            xx = (float)c*antiscale;
            yy = (float)r*antiscale;

            nx0 = floor(xx+EPS);
            nx1 = ceil(xx+EPS);
            ny0 = floor(yy+EPS);
            ny1 = ceil(yy+EPS);

            nx0 = min(max(nx0, 0), cols-1);
            nx1 = min(max(nx1, 0), cols-1);
            ny0 = min(max(ny0, 0), rows-1);
            ny1 = min(max(ny1, 0), rows-1);

            weight1 = (nx1-xx)*(ny1-yy);  // upper left
            weight2 = (xx-nx0)*(ny1-yy);  // upper right
            weight3 = (nx1-xx)*(yy-ny0);  // lower left
            weight4 = (xx-nx0)*(yy-ny0);  // lower right

            for (int k = 0; k < chan; ++k) { 
                int i1 = linear_index(ny0, nx0, k, cols, chan);
                int i2 = linear_index(ny1, nx0, k, cols, chan);
                int i3 = linear_index(ny0, nx1, k, cols, chan);
                int i4 = linear_index(ny1, nx1, k, cols, chan);

                int i_out = linear_index(r, c, k, out_cols, chan);

                out[i_out] = 
                    weight1*img[i1] + 
                    weight2*img[i2] + 
                    weight3*img[i3] + 
                    weight4*img[i4];
            }
        }
    }
}

void Slic::build_pyramid(const vector<unsigned char>& img, int levels) {
#ifdef SLIC_DEBUG
    unsigned long t1,t2;
    t1 = now_us();
    printf("Slic::build_pyramid ");
#endif    
    // build the pyramid for the image data
    pyramid = vector<vector<unsigned char> >(levels);
    int n = rows[0]*cols[0]*chan;
    pyramid[0] = vector<unsigned char>(n);
    copy( img.begin(), img.begin()+n, pyramid[0].begin() );

    for (int i = 1; i < levels; ++i) {
        imresize(pyramid[i], pyramid[i-1], rows[i-1], cols[i-1], rows[i], cols[i]);
    }
    // build the pyramid for assignments:
    assignments_pyramid = vector<vector<int> >(levels);
    for (int i = 0; i < levels; ++i) { 
        assignments_pyramid[i] = vector<int>(rows[i]*cols[i],0);
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", (t2-t1)/1000.0 );
#endif
}

void Slic::upsample_cluster_centers(int level) {
    // level is the current level.
    if (level == 0) { printf("Should not be called with this argument!\n"); }
    float s1 = ((float)rows[level-1])/((float)rows[level]);
    float s2 = ((float)cols[level-1])/((float)cols[level]);
    float s = (s1+s2)/2.0;
    // This needs to be done when we go up on the pyramid levels.
    for (int k = 0; k < K; ++k) { 
        centers[k].x = centers[k].x * s;
        centers[k].y = centers[k].y * s;
    }
}

void Slic::upsample_assignments(int level) {
#ifdef SLIC_DEBUG
    unsigned long t1, t2;
    t1 = now_us();
    printf("Slic::upsample_assignments ");
#endif
    // level is the current level.
    if (level == 0) { printf("Should not be called with this argument!\n"); }
    vector<int> *assignments = &assignments_pyramid[level];
    vector<int> *assignments_out = &assignments_pyramid[level-1];
    vector<unsigned char> *data = &pyramid[level-1];
    int rows_ = rows[level];
    int cols_ = cols[level];
    int rows_big = rows[level-1];
    int cols_big = cols[level-1];
    
    float EPS = 1e-4; 
    float xx, yy;  // location and flow directions
    int nx0, nx1, ny0, ny1;  // neighboring location
    int i1, i2, i3, i4, i_out, i_;
    int ass1, ass2, ass3, ass4, temp;
    int d1, d2, d3, d4;
 
    float scale = (float(rows_big)/float(rows_)+float(cols_big)/float(cols_))/2.0;
    float antiscale = 1.0/scale;
    int Mp = get_distance_weight(level-1);
    for (size_t r = 0; r < (size_t)rows_big; ++r) {
        for (size_t c = 0; c < (size_t)cols_big; ++c) {
            xx = (float)c*antiscale;
            yy = (float)r*antiscale;

            nx0 = floor(xx+EPS);
            nx1 = ceil(xx+EPS);
            ny0 = floor(yy+EPS);
            ny1 = ceil(yy+EPS);

            nx0 = min(max(nx0, 0), cols_-1);
            nx1 = min(max(nx1, 0), cols_-1);
            ny0 = min(max(ny0, 0), rows_-1);
            ny1 = min(max(ny1, 0), rows_-1);

            i1 = linear_index(ny0, nx0, cols_);
            i2 = linear_index(ny1, nx0, cols_);
            i3 = linear_index(ny0, nx1, cols_);
            i4 = linear_index(ny1, nx1, cols_);
            i_out = linear_index(r, c, cols_big);

            ass1 = (*assignments)[i1];
            ass2 = (*assignments)[i2];
            ass3 = (*assignments)[i3];
            ass4 = (*assignments)[i4];

            if ( (ass1==ass2) && (ass2==ass3) && (ass3==ass4) ) {
                // we are on the interior of the superpixel (whew)
                (*assignments_out)[i_out] = ass1;
            } else {
                d1 = d2 = d3 = d4 = 0;
                for (int k = 0; k < chan; ++k) { 
                    i_ = linear_index(r, c, k, cols_big, chan);
                    temp = (*data)[i_]-centers[ass1].rgb[k];
                    d1 += temp*temp;
                    temp = (*data)[i_]-centers[ass2].rgb[k];
                    d2 += temp*temp;
                    temp = (*data)[i_]-centers[ass3].rgb[k];
                    d3 += temp*temp;
                    temp = (*data)[i_]-centers[ass4].rgb[k];
                    d4 += temp*temp;    
                }
                temp = (r-centers[ass1].y)*(r-centers[ass1].y) + 
                    (c-centers[ass1].x)*(c-centers[ass1].x);
                d1 += Mp*temp;
                temp = (r-centers[ass2].y)*(r-centers[ass2].y) + 
                    (c-centers[ass2].x)*(c-centers[ass2].x);
                d2 += Mp*temp;
                temp = (r-centers[ass3].y)*(r-centers[ass3].y) + 
                    (c-centers[ass3].x)*(c-centers[ass3].x);
                d3 += Mp*temp;
                temp = (r-centers[ass4].y)*(r-centers[ass4].y) + 
                    (c-centers[ass4].x)*(c-centers[ass4].x);
                d4 += Mp*temp;
                
                if ((d1 <= d2) && (d1 <= d2) && (d1 <= d3)) { 
                    (*assignments_out)[i_out] = ass1;
                } else if ((d2 <= d1) && (d2 <= d3) && (d2 <= d4)) { 
                    (*assignments_out)[i_out] = ass2;
                } else if ((d3 <= d1) && (d3 <= d2) && (d3 <= d4)) { 
                    (*assignments_out)[i_out] = ass3;
                } else { 
                    (*assignments_out)[i_out] = ass4;
                }
            }
        }
    }

#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", (t2-t1)/1000.0 );
#endif
}

void Slic::get_superpixel_pixcount(int level) {
    int cols_ = cols[level];
    int rows_ = rows[level];
    int n = rows_*cols_;

    pixcount = vector<int>(K, 0);

    vector<int> *assignments = &assignments_pyramid[level];
    for (size_t i = 0; i < n; ++i) { 
        pixcount[ (*assignments)[i] ]++;
    }
}

void Slic::print_assignment() {
    // this is useful for piping output, e.g. main > assignment.txt
    int n = rows[0]*cols[0];
    for (int i=0; i<n; ++i)
        printf("%d\n", assignments_pyramid[0][i]);
}

void Slic::print_centers() {
    for (int k=0; k<K; ++k)
        printf("%d %d\n", centers[k].x, centers[k].y );
}

void Slic::write_assignment(const char* fname) {
    FILE* fid;
    fid = fopen(fname,"w");
    if (fid == NULL) {
        printf("Could not open %s for writing\n", fname);
        return;
    }
    int n = rows[0]*cols[0];
    for (int i=0; i<n; ++i)
        fprintf(fid, "%d\n", assignments_pyramid[0][i]);
    fclose(fid);
}

void Slic::write_centers(const char* fname) {
    FILE* fid;
    fid = fopen(fname,"w");
    if (fid == NULL) {
        printf("Could not open %s for writing\n", fname);
        return;
    }
    for (int k=0; k<K; ++k)
        fprintf(fid, "%d %d\n", centers[k].x, centers[k].y);
    fclose(fid);
}

// ! Returns an RGB image with highlighted superpixel boundaries
vector<unsigned char> Slic::get_boundary_image(
        const vector<unsigned char>& rgb_data, int level) {
    int rows_ = rows[level];
    int cols_ = cols[level];
    vector<int> *assignments = &assignments_pyramid[level];
    
    int k, k_left=0, k_right=0, k_bottom=0, k_top=0;
    float w;
    vector<unsigned char> img = vector<unsigned char>(rows_*cols_*chan);
    for (int c=0; c<cols_; ++c) {
        for (int r=0; r<rows_; ++r) {
            int i = linear_index(r, c, cols_);
            k = (*assignments)[i];
            if (c>0) {
                k_left = (*assignments)[ linear_index(r, c-1, cols_) ];
            } else { k_left = -1; }
            if (c < cols_-1) {
                k_right = (*assignments)[ linear_index(r, c+1, cols_) ];
            } else { k_right = -1; }
            if (r>0) {
                k_top = (*assignments)[ linear_index(r-1, c, cols_) ];
            } else { k_top = -1; }
            if (r < rows_-1) {
                k_bottom = (*assignments)[ linear_index(r+1, c, cols_) ];      
            } else { k_bottom = -1; }

            w = ((float)(k==k_right) + (float)(k==k_left) + 
                 (float)(k==k_bottom) + (float)(k==k_top))/4.0;
            if (chan == 3) {
                for (int u = 0; u < 3; ++u) {
                    i = linear_index(r,c, u, cols_, 3);
                    img[i] = rgb_data[i]*w;
                }
            } else {
                img[i] = rgb_data[i]*w;
            }
        }
    }
    return img;
}
