//! @file tSlic.cpp
#include "tSlic.h"
//#include "bcv_opencv.cpp"

tSlic::tSlic() {
}

tSlic::~tSlic() {
}

tSlic::tSlic(tslic_params& p) {
#ifdef SLIC_DEBUG
    ulong t1, t2;
    t1 = now_us();
    printf("tSlic::constructor ");
#endif

    check_parameters(p);

    // infer *actual* number of superpixels.
    float ar = float(p.cols)/float(p.rows);
    nx = sqrt( ar*float(p.K) );
    ny = sqrt( float(p.K)/ar );
    K = nx*ny;
    Knom = K;
    // this is the number of pixels that we will initialize with.
    // in practice, it could be that due to scene motion, we need to increase
    // the number of superpixels (think of zoom-out motions - existing regions
    // are grouped together, and new regions appear). so, we have an upper
    // bound on the total number of possible superpixels:
    Kmax = 2*K;
    num_voxels_lived = 0; // this is updated whenever result is processed.

    // figure out the size of each superpixel:
    float step_x = float(p.cols) / float(nx+1.0);
    float step_y = float(p.rows) / float(ny+1.0);
    //
    float nom_area = step_x * step_y;

    min_area = max(2.0f, nom_area * p.min_area_frac);
    max_area = min((float)p.cols*p.rows, nom_area * p.max_area_frac);
    printf("min_area: %d, max_area: %d\n", min_area, max_area);
    //-------------------------------------------------------------------------
    rows = p.rows;
    cols = p.cols;
    chan = p.chan; // marshall
    search_region_x = round(step_x*2.0);
    search_region_y = round(step_y*2.0);

    int n = p.rows*p.cols;
    d = vector<int>(n, INT_MAX);
    assignments = vector<int>(n, 0);
    pixcount = vector<int>(Kmax, 0);
    M = p.M * (chan); // change M with number of channels..
    num_iters = p.num_iters;

#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("took %f\n", (t2-t1)/1000.0);
#endif
}

void tSlic::segment(const vector<uchar>& img, const vector<int>& pts) {
    // update image...
    img_data = img;

    if (num_voxels_lived == 0) {
        // this is executed only once.
        init_centers();
        adjust_centers(); 
    } else {
        if (pts.size() > 0) { 
            warp_centers_with_tracks(pts, img);
        }
        assign();
        int nsplit = split_move();
        assign();
        //if (nsplit > 0) { assign(); } // may not be necessary to do it here!
        int nmerge = merge_move();
        //if (nmerge > 0) { assign(); }
        assign();
        remove_empty_superpixels();
    }
    // main Slic block:
    for (int i=0; i<num_iters; ++i) {
        assign();
        update();
    }
    printf("contiguity\n");
    ensure_contiguity();
    printf("remove empty\n");
    remove_empty_superpixels();

    increment_lifetime();

    if (num_voxels_lived == 0) {
        num_voxels_lived = K;
    }

    // some statistics:
    /*
    for (int i = 0; i < K; ++i) { 
        int bad1 = (is_cluster_invalid(
        return ids;centers[i]));
        int bad2 = pixcount[i] == 0;
        if (bad1 | bad2) { 
            printf("k=%d, num=%d, id=%d\n", i, pixcount[i], centers[i].id );
        }
    }*/
}


// this is executed once, before the first segmentation.
void tSlic::init_centers() {
    #ifdef SLIC_DEBUG
    printf("tSlic::init_centers()\n");
    #endif
    centers = vector<tslic_cluster>(Kmax);

    float step_x = float(cols) / float(nx+1.0);
    float step_y = float(rows) / float(ny+1.0);
    float offset_x = step_x / 2.0;
    float offset_y = step_y / 2.0;
 
    // initialize all centers to bogus values:
    for (int i = 0; i < Kmax; ++i) {
        make_cluster_invalid(centers[i]);
    }
    int i=0;
    for (int x=0; x<nx; ++x) {
        for (int y=0; y<ny; ++y) { 
            centers[i].id = i;
            centers[i].x = (x)*step_x+offset_x;
            centers[i].y = (y)*step_y+offset_y;
            for (int u=0; u<chan; ++u) {
                int k = linear_index(y, x, u, cols, chan);
                centers[i].color[u] = img_data[k];
            }
            i++;
        }
    }
}

void tSlic::adjust_centers() {
#ifdef SLIC_DEBUG
    printf("tSlic::adjust_centers ");
    unsigned long t1,t2;
    t1 = now_us();
#endif
    for (int k=0; k < K; ++k) { 
        int x = centers[k].x;
        int y = centers[k].y;
        
        // force centers to be away from boundaries
        if (x < 2) { x = 2; }
        if (y < 2) { y = 2; }
        if (x > cols-2) { x = cols-2; }
        if (y > rows-2) { y = rows-2; }

        int min_val = INT_MAX;
        int best_x=x;
        int best_y=y;

        for (int yy=y-1; yy<=(y+1); yy++) {
            for (int xx=x-1; xx<=(x+1); xx++) {
                float grad = 0.0;
                int i_bot = linear_index(yy+1, xx, 0, cols, chan);
                int i_top = linear_index(yy-1, xx, 0, cols, chan);
                int i_lef = linear_index(yy, xx-1, 0, cols, chan);
                int i_rig = linear_index(yy, xx+1, 0, cols, chan);
                for (int u=0; u<chan; ++u) {
                    float v = (int)img_data[i_bot+u]-(int)img_data[i_top+u];
                    float h = (int)img_data[i_lef+u]-(int)img_data[i_rig+u];
                    grad += v*v + h*h;
                }
                if (grad < min_val) {
                    min_val = grad;
                    best_x = xx;
                    best_y = yy;
                }
            }
        }
        // update cluster center
        centers[k].x = best_x;
        centers[k].y = best_y;
        int i = linear_index(best_y, best_x, 0, cols, chan);
        for (int u = 0; u < chan; ++u) { 
            centers[k].color[u] = img_data[i+u];
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}


void tSlic::assign() {
#ifdef SLIC_DEBUG
    printf("tSlic::assign ");
    unsigned long t1,t2;
    t1 = now_us();
#endif
    int n = rows*cols;
    //for (int i=0; i<n; ++i) { d[i] = INT_MAX; }
    fill(d.begin(), d.end(), INT_MAX); // should be faster.

    for (int k=0; k<K; ++k) {
        if (is_cluster_invalid(centers[k])) { continue; } 
        int xc = centers[k].x;
        int yc = centers[k].y;   
        
        int xlo = max(0, xc-search_region_x );
        int xhi = min(cols-1, xc+search_region_x );
        int ylo = max(0, yc-search_region_y );
        int yhi = min(rows-1, yc+search_region_y );

        for (int y=ylo; y<=yhi; ++y) {
            int ddy = M*(y-yc)*(y-yc);
            // from center down:
            for (int x=xlo; x<=xhi; ++x) {
                int i = linear_index(y, x, cols);
                //if (ddy >= d[i])
                //    continue;
                int dist = M*(x-xc)*(x-xc) + ddy;                
                if (dist >= d[i])
                    continue;                
                
                int q = linear_index(y, x, 0, cols, chan);
                for (int u = 0; u < chan; ++u) { 
                    int temp = ((int)img_data[q+u]-(int)centers[k].color[u]);
                    dist += temp*temp;
                }
                if (dist < d[i]) {
                    d[i] = dist;
                    assignments[i] = k;  
                }
            }
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}

void tSlic::update() {
#ifdef SLIC_DEBUG
    printf("Slic::update ");
    unsigned long t1,t2; 
    t1 = now_us();
#endif
    int color[3];
    for (int k=0; k<K; ++k) {
        if (is_cluster_invalid(centers[k])) { continue; } 
        int xc = centers[k].x;
        int yc = centers[k].y;   

        int xlo = max(0, xc-search_region_x);
        int xhi = min(cols-1, xc+search_region_x);
        int ylo = max(0, yc-search_region_y);
        int yhi = min(rows-1, yc+search_region_y);

        int nums=0;
        color[0] = color[1] = color[2] = 0;
        int sumx=0;
        int sumy=0;
        for (int y=ylo; y<=yhi; ++y) {
            for (int x=xlo; x<=xhi; ++x) {
                int i = linear_index(y, x, cols);
                int q = linear_index(y, x, 0, cols, chan);
                if (assignments[i]==k) { 
                    for (int u = 0; u < chan; ++u) { 
                        color[u] += (int)img_data[q+u];
                    }
                    sumx += x;
                    sumy += y;
                    nums++;
                }
            }
        }
        if (nums>0) {
            centers[k].x = round( float(sumx) / float(nums) ); 
            centers[k].y = round( float(sumy) / float(nums) );
            for (int u = 0; u < chan; ++u) { 
                centers[k].color[u] = round( float(color[u]) / float(nums) );
            }
        } else { make_cluster_invalid(centers[k]); }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}

void tSlic::increment_lifetime() {
    for (int k = 0; k < K; ++k) { 
        centers[k].lifetime++;
    }
}

void tSlic::get_superpixel_pixcount() {
    pixcount = vector<int>(Kmax, 0);
    for (int i = 0; i < assignments.size(); ++i) { 
        // note: this should never really happen. that this happens is a symptom
        // of something worse.
        if ((assignments[i] <0 ) || (assignments[i]>=Kmax)) { continue; }
        pixcount[ assignments[i] ]++;
    }
}

int tSlic::merge_move() {
#ifdef SLIC_DEBUG
    printf("tSlic::merge_move: ");
    ulong t1 = now_us();
#endif
    int num_merged = 0;
    get_superpixel_pixcount();
    for (int k = 0; k < K; ++k) { 
        if (is_cluster_invalid(centers[k])) { continue; }
        if (pixcount[k] < min_area) {
            // mark for removal...
            make_cluster_invalid(centers[k]);
            num_merged++;
        }
    }
#ifdef SLIC_DEBUG
    printf("merged %d ", num_merged);
    printf("elapsed: %f ms\n", (now_us()-t1)/1000.0f );
#endif
    return num_merged;
}

int tSlic::split_move() {
#ifdef SLIC_DEBUG
    printf("tSlic::split_move: ");
    ulong t1 = now_us();
#endif
    int num_split = 0;
    int Know = K;
    get_superpixel_pixcount();
    for (int k = 0; k < Know; ++k) { 
        if (K >= Kmax) {
            printf("Cant add anymore superpixels.\n");
            break; // cant add anymore superpixels
        }
        if (is_cluster_invalid(centers[k])) { continue; }
        if (pixcount[k] > max_area) {
            
            // needs to be split... 
            num_split++;
            vector<float> sigma = compute_superpixel_variance(k);
            vector<int> bds = compute_superpixel_bounds(k); 
            
            int n1 = 0;
            int n2 = 0;
            int set_id;
            int xc_x = centers[k].x;
            int xc_y = centers[k].y;

            centers[k].x = 0; //x;
            centers[k].y = 0; //y1;
            centers[K].x = 0; //x;
            centers[K].y = 0; //y2;
            for (int u = 0; u < chan; ++u) {
                centers[k].color[u] = 0;
                centers[K].color[u] = 0;
            }
            vector<int> color1 = vector<int>(3, 0);
            vector<int> color2 = vector<int>(3, 0);
            
            vector<float> eigvals = eigvals_2x2(sigma[0], sigma[1], sigma[2], sigma[3]);
            vector<float> eigvec = eigvec_2x2(sigma[0], sigma[1], sigma[2], sigma[3], eigvals[0]);
            
            // update the assignments
            for (int yy = bds[2]; yy <= bds[3]; ++yy) { 
                for (int xx = bds[0]; xx <= bds[1]; ++xx) { 
                    int ids = linear_index(yy, xx, cols);
                    if (assignments[ids] != k) { continue; }
                    int ids2 = linear_index(yy, xx, 0, cols, chan);
                    int val = (xx-xc_x)*eigvec[0] + (yy-xc_y)*eigvec[1];
                    
                    if (val > 0) {
                        set_id = k;
                        n1++;
                        for (int u = 0; u < chan; ++u) {
                            color1[u] += img_data[ids2+u];
                        }
                    } else { 
                        set_id = K;
                        n2++;
                        for (int u = 0; u < chan; ++u) {
                            color2[u] += img_data[ids2+u];
                        }
                    }
                    assignments[ids] = set_id;
                    centers[set_id].x += xx;
                    centers[set_id].y += yy;
                }
            }
           
            centers[k].x /= (float)max(n1, 1);
            centers[k].y /= (float)max(n1, 1);
            centers[K].x /= (float)max(n2, 1);
            centers[K].y /= (float)max(n2, 1);
            for (int u =0; u < chan; ++u) { 
                centers[k].color[u] = ((float)color1[u]/(float)max(n1, 1));
                centers[K].color[u] = ((float)color2[u]/(float)max(n2, 1));
            }
            
            centers[K].id = num_voxels_lived; 
            centers[K].lifetime = 0;
            num_voxels_lived++;
            K++;

            // reupdate pixcount, since we messed with it here
            get_superpixel_pixcount();
        }
    }
#ifdef SLIC_DEBUG
    printf("split %d ", num_split);
    printf("elapsed: %f ms\n", (now_us()-t1)/1000.0f );
#endif
    return num_split;
}

//! Returns a 2x2 covariance matrix associated with the superpixel
vector<float> tSlic::compute_superpixel_variance(int k) { 
    vector<float> Sigma = vector<float>(4, 0.0);
    int xc = centers[k].x;
    int yc = centers[k].y;

    int xlo = max(0, xc-search_region_x );
    int xhi = min(cols-1, xc+search_region_x );
    int ylo = max(0, yc-search_region_y );
    int yhi = min(rows-1, yc+search_region_y );

    int N = 0;
    for (int y=ylo; y<=yhi; ++y) {
        for (int x=xlo; x<=xhi; ++x) {
            int i = linear_index(y, x, cols);
            if (assignments[i] == k) {
                Sigma[0] += (x-xc)*(x-xc);
                Sigma[1] += (x-xc)*(y-yc);
                Sigma[2] += (x-xc)*(y-yc);
                Sigma[3] += (y-yc)*(y-yc);
                N++;
            }  
        }
    }
    for (int q = 0; q < 4; ++q) { Sigma[q] /= float(N); }
    return Sigma;  
}

//! Returns a 2x2 upper/lower bounds of the superpixel
vector<int> tSlic::compute_superpixel_bounds(int k) { 
    vector<int> Q = vector<int>(4);
    Q[0] = Q[2] = numeric_limits<int>::max();
    Q[1] = Q[3] = numeric_limits<int>::min();

    int xc = centers[k].x;
    int yc = centers[k].y;

    int xlo = max(0, xc-search_region_x );
    int xhi = min(cols-1, xc+search_region_x );
    int ylo = max(0, yc-search_region_y );
    int yhi = min(rows-1, yc+search_region_y );

    for (int y=ylo; y<=yhi; ++y) {
        for (int x=xlo; x<=xhi; ++x) {
            int i = linear_index(y, x, cols);
            if (assignments[i] == k) {
                Q[0] = min(Q[0], x);
                Q[1] = max(Q[1], x);
                Q[2] = min(Q[2], y);
                Q[3] = max(Q[3], y);
            }
        }
    }
    return Q;
}



void tSlic::ensure_contiguity() {
#ifdef SLIC_DEBUG
    unsigned long t1, t2;
    t1 = now_us();
    printf("Slic::ensure_contiguity ");
#endif
    int n = rows*cols;
    vector<bool> mask = vector<bool>(n);
    vector<bool> visited = vector<bool>(n);
    for (int k=0; k<K; ++k) {
        int xc = centers[k].x;
        int yc = centers[k].y;
        if (is_cluster_invalid(centers[k])) { continue; }

        int xlo = max(0, xc-search_region_x);
        int xhi = min(cols-1, xc+search_region_x);
        int ylo = max(0, yc-search_region_y);
        int yhi = min(rows-1, yc+search_region_y);
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
        int i = linear_index(yc, xc, cols);
        if ( assignments[i]==k ) { 
            // yay!
            origin_x = xc;
            origin_y = yc;
            nums = 1;
        } else {
            for (int y=ylo; y<=yhi; ++y) {
                int ddy = (y-yc)*(y-yc);
                if (ddy > dist_to_origin) { continue; }
                for (int x=xlo; x<=xhi; ++x) {
                    int i = linear_index(y, x, cols);
                    if (assignments[i]==k) { 
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
                int i = linear_index(y, x, cols);
                visited[i] = false;
                mask[i] = (assignments[i] == k);
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
            centers[k].id = -1; // make invalid
            continue;
            // this is a can of worms, but it is handled later
        }
        // we know the point closest to center that belongs to this cluster,
        // so start here and try to 'unset' all the 'set' pixels in the mask.
        try_fill(origin_x, origin_y, linear_index(origin_y, origin_x, cols),
                mask, visited, xlosp, xhisp, ylosp, yhisp);

        int num_left = 0;
        for (int y=ylosp; y<=yhisp; ++y) {
            for (int x=xlosp; x<=xhisp; ++x) {
                int i = linear_index(y, x, cols);
                if (mask[i]) {
                    num_left += 1;
                    break;
                }
            }
            if (num_left > 0) { break; }
        }

        if (num_left > 0) {
            // assign the remaining pixels to neighboring clusters
            reassign_neighbors(mask, k, xlosp, xhisp, ylosp, yhisp);
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}


void tSlic::reassign_neighbors(vector<bool>& mask, 
        int cluster, int xlo, int xhi, int ylo, int yhi) {
    vector<int> stack;
    stack.reserve(1000);
    // need to reassign pixels.
    for (int y=ylo; y<=yhi; ++y) {
        for (int x=xlo; x<=xhi; ++x) {
            int i = linear_index(y, x, cols);
            if (!mask[i]) {
                continue;
            } else {
                mask[i] = false;
            }                    
            int i_lef = linear_index(y, x-1, cols);
            int i_rig = linear_index(y, x+1, cols);
            int i_top = linear_index(y-1, x, cols);
            int i_bot = linear_index(y+1, x, cols);

            int neighbor=-1;
            if ((x>0) && ( assignments[i_lef] != cluster))
                neighbor = assignments[i_lef];
            else if ((x<cols-1) && ( assignments[i_rig] != cluster ))
                neighbor = assignments[i_rig];
            else if ((y>0) && ( assignments[i_top] != cluster ))
                neighbor = assignments[i_top];
            else if ((y<rows-1) && ( assignments[i_bot] != cluster))
                neighbor = assignments[i_bot];
            if (neighbor >= 0) {
                // empty the stack...
                assignments[i] = neighbor;
                while (!stack.empty()) {
                    int s = stack[0];
                    stack.erase( stack.begin() );
                    assignments[s] = neighbor;
                }
            } else {
                // push pixel onto the stack...
                stack.push_back( i );
            }
        }
    }
}

void tSlic::try_fill(int x, int y, int i,
        vector<bool>& mask, vector<bool>& visited,
        int xlo, int xhi, int ylo, int yhi) {

    visited[i] = true;
    bool mask_val = mask[i];
    mask[i] = false;
    // strict inequalities here, shouldnt go out-of-bounds.
    if (mask_val) {
        // go left
        int i_lef = linear_index(y, x-1, cols);
        if ((x > xlo) && (visited[i_lef] == false))
            try_fill(x-1, y, i_lef, mask, visited, xlo, xhi, ylo, yhi);
        // go right
        int i_rig = linear_index(y, x+1, cols);
        if ((x < xhi) && (visited[i_rig] == false))
            try_fill(x+1, y, i_rig, mask, visited, xlo, xhi, ylo, yhi);
        // go bottom
        int i_bot = linear_index(y+1, x, cols);
        if ((y < yhi) && (visited[i_bot] == false))
            try_fill(x, y+1, i_bot, mask, visited, xlo, xhi, ylo, yhi);
        // go top
        int i_top = linear_index(y-1, x, cols);
        if ((y > ylo) && (visited[i_top] == false))
            try_fill(x, y-1, i_top, mask, visited, xlo, xhi, ylo, yhi);
        //int i_lu = linear_index(y-1, x-1, cols_);
        //if ((y > ylo) && (x > xlo) && (visited[i_lu] == false))
        //    try_fill(x-1, y-1, i_lu, mask, visited, xlo, xhi, ylo, yhi, cols_);

        //int i_ru = linear_index(y-1, x+1, cols_);
        //if ((y > ylo) && (x < xhi) && (visited[i_ru] == false))
        //    try_fill(x+1, y-1, i_ru, mask, visited, xlo, xhi, ylo, yhi, cols_);

        //int i_lb = linear_index(y+1, x-1, cols_);
        //if ((y < yhi) && (x > xlo) && (visited[i_lb] == false))
        //    try_fill(x-1, y+1, i_lb, mask, visited, xlo, xhi, ylo, yhi, cols_);

        //int i_rb = linear_index(y+1, x+1, cols_); 
        //if ((y < yhi) && (x < xhi) && (visited[i_rb] == false))
        //    try_fill(x+1, y+1, i_rb, mask, visited, xlo, xhi, ylo, yhi, cols_);
    }
}

void tSlic::remove_empty_superpixels() {
#ifdef SLIC_DEBUG
unsigned long t1 = now_us();
#endif
    // clean-up routine.
    printf("getting pixcount\n");
    get_superpixel_pixcount();

    int numlost = 0;
    vector<tslic_cluster> temp_centers = vector<tslic_cluster>();
    temp_centers.reserve(K);
    printf("making temp centers\n");
    for (int i = 0; i < K; ++i) {
        if (is_cluster_invalid(centers[i]) || (pixcount[i]==0)) {
            numlost++;
        } else {
            temp_centers.push_back( centers[i] );
        }
    }
    printf("created tempclusters\n");
    // lucky case.
    if (numlost == 0) { return; }
   
    // copy the 'good' clusters 
    for (int i = 0; i < temp_centers.size(); ++i) { 
        centers[i] = temp_centers[i];
    }
    printf("copied clusters\n");
    // invalidate the rest
    for (int i = temp_centers.size(); i < centers.size(); ++i) { 
        make_cluster_invalid( centers[i] );
    }

    K = K - numlost;
    printf("K = %d (numlost = %d)\n", K, numlost);
#ifdef SLIC_DEBUG
    printf("tSlic::remove_empty_superpixels elapsed: %f\n", (now_us()-t1)/1000.0f );
#endif
}

//! Points is a vector of predicted cluster center locations, ordered as:
//! (x,y), (x,y), ...
void tSlic::warp_centers_with_tracks(const vector<int>& pts, const vector<uchar>& img) {
    int x, y; 
    for (int k = 0; k < K; ++k) { 
        centers[k].x = pts[2*k];
        centers[k].y = pts[2*k+1];
        x = centers[k].x;
        y = centers[k].y;
        // update cluster color
        // Here we may encounter a situation where a superpixel moves
        // entirely outside of the screen. It is not clear what to do in
        // this case. ChangCVPR13 discuss this (see supplement).
        // For now, if a superpixel moves outside, mark it as invalid...
        // TODO: UGH MAKE THIS INTO A MACRO
        if (!is_in_range(x,y,0)) {
            make_cluster_invalid(centers[k]);
        } else {
            int i = linear_index(y, x, 0, cols, chan);
            for (int u = 0; u < chan; ++u) { 
                centers[k].color[u] = img[i+u];
            }
        }
    }
}

void tSlic::check_parameters(tslic_params& p) {
    assert( (p.rows>0) && (p.cols>0) && (p.chan>0) 
            && "tslic parameters are not set correctly.");
    assert( (p.K>0) && (p.M>0) && (p.num_iters>0)
            && "tslic parameters are not set correctly.");
    assert( (p.min_area_frac>0) && (p.max_area_frac>0) 
            && "tslic parameters are not set correctly.");
}

vector<tslic_node> tSlic::construct_graph() {
#ifdef SLIC_DEBUG
    unsigned long t1,t2;
    t1 = now_us();
    printf("Slic::construct_graph ");
#endif
    // construct adjacency matrix:
    vector<set<int> > adj = vector< set<int> >(K);
    for (int y=1; y<rows-1; ++y) {
        for (int x=1; x<cols-1; ++x) {    
            int i = assignments[ linear_index(y, x, cols) ];
            int i_rig = assignments[ linear_index(y, x+1, cols) ];
            int i_bot = assignments[ linear_index(y+1, x, cols) ];

            if (i != i_rig) {
                adj[i].insert(i_rig);
                adj[i_rig].insert(i);
            }
            if (i != i_bot) {
                adj[i].insert(i_bot);
                adj[i_bot].insert(i);
            }
        }
    }

    vector<tslic_node> graph = vector<tslic_node>(K);
    // now start building the graph:
    for (int i=0; i<K; ++i) {
        // put in neighbors
        graph[i].neighbors = vector<int>( adj[i].size() );
        int u=0;
        for (set<int>::iterator it=adj[i].begin(); it!=adj[i].end(); ++it) {
            graph[i].neighbors[u] = *(it);
            u++;
        }
        //
        graph[i].data = centers[i];
        graph[i].pixels_linear = vector<int>();
        graph[i].pixels_xy = vector<pair<int,int> >();
        graph[i].pixels_linear.reserve(search_region_x*search_region_y);
        graph[i].pixels_xy.reserve(search_region_x*search_region_y);
    }        

    for (int y = 0; y < rows; ++y) { 
        for (int x = 0; x < cols; ++x) {
            int k = assignments[ linear_index(y,x,cols) ];
            graph[k].pixels_xy.push_back( pair<int,int>(x,y) );
            graph[k].pixels_linear.push_back( linear_index(y,x,cols) );
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
    return graph;
}

bool inline tSlic::is_cluster_invalid(tslic_cluster& c) { 
    return (c.id < 0);
}

void inline tSlic::make_cluster_invalid(tslic_cluster& c) {
    c.x = -1;
    c.y = -1;
    c.color[0] = 0;
    c.color[1] = 0;
    c.color[2] = 0;
    c.id = -1;
    c.lifetime = 0;
}

bool inline tSlic::is_in_range(int x, int y, int d) {
    return ((x >= d) && (x <= cols-1-d) && (y >= d) && (y <= rows-1-d));
}

/* ! Returns an RGB image with highlighted superpixel boundaries */
vector<uchar> tSlic::get_boundary_image(
        const vector<uchar>& rgb_data, int rows, int cols, int chan) {
    if ((rows==0) || (cols==0) || (chan==0)) { // use default
        rows = this->rows;
        cols = this->cols;
        chan = this->chan;
    }
    int k, k_left=0, k_right=0, k_bottom=0, k_top=0;
    float w;
    vector<uchar> img = vector<uchar>(rows*cols*chan);
    for (int c=0; c<cols; ++c) {
        for (int r=0; r<rows; ++r) {
            int i = linear_index(r, c, cols);
            k = assignments[i];
            if (c>0) {
                k_left = assignments[ linear_index(r, c-1, cols) ];
            } else { k_left = -1; }
            if (c < cols-1) {
                k_right = assignments[ linear_index(r, c+1, cols) ];
            } else { k_right = -1; }
            if (r>0) {
                k_top = assignments[ linear_index(r-1, c, cols) ];
            } else { k_top = -1; }
            if (r < rows-1) {
                k_bottom = assignments[ linear_index(r+1, c, cols) ];      
            } else { k_bottom = -1; }

            w = ((float)(k==k_right) + (float)(k==k_left) + 
                    (float)(k==k_bottom) + (float)(k==k_top))/4.0;
            i = linear_index(r, c, 0, cols, chan);
            for (int u = 0; u < chan; ++u) {
                img[i+u] = rgb_data[i+u]*w;
            }
        }
    }

    return img;
}

vector<int> tSlic::get_ids() {
    vector<int> ids = vector<int>(K);
    for (int k = 0; k < K; ++k) { ids[k] = centers[k].id; }
    return ids;
}
