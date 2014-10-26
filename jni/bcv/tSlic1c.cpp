//! @file tSlic1c.cpp
#include "tSlic1c.h"

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
    search_region_x = round(step_x*2.0);
    search_region_y = round(step_y*2.0);

    int n = p.rows*p.cols;
    d = vector<ushort>(n, USHRT_MAX); 
    assignments = vector<ushort>(n, 0);
    mask = vector<uchar>(n);
    visited = vector<uchar>(n);
    pixcount = vector<int>(Kmax, 0);
    M = p.M;
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
    ensure_contiguity();
    remove_empty_superpixels();

    increment_lifetime();

    if (num_voxels_lived == 0) {
        num_voxels_lived = K;
    }
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
    for (int y=0; y<ny; ++y) { 
        for (int x=0; x<nx; ++x) {
            centers[i].id = i;
            centers[i].x = (x)*step_x+offset_x;
            centers[i].y = (y)*step_y+offset_y;
            centers[i].color = img_data[ linear_index(y, x, cols) ];
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
                int i_bot = linear_index(yy+1, xx, cols);
                int i_top = linear_index(yy-1, xx, cols);
                int i_lef = linear_index(yy, xx-1, cols);
                int i_rig = linear_index(yy, xx+1, cols);
                
                int v = (int)img_data[i_bot]-(int)img_data[i_top];
                int h = (int)img_data[i_lef]-(int)img_data[i_rig];
                grad = v*v + h*h;
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
        centers[k].color = img_data[ linear_index(best_y, best_x, cols) ];
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
    fill(d.begin(), d.end(), INT_MAX); // should be faster.

    for (int k=0; k<K; ++k) {
        if (is_cluster_invalid(centers[k])) { continue; } 
        int xc = centers[k].x;
        int yc = centers[k].y;   
        
        int xlo = max(0, xc-search_region_x );
        int xhi = min(cols-1, xc+search_region_x );
        int ylo = max(0, yc-search_region_y );
        int yhi = min(rows-1, yc+search_region_y );

        int y,x,i;
        for (y=ylo; y<=yhi; ++y) {
            int ddy = M*(y-yc)*(y-yc);
            // from center down:
            for (x=xlo, i=linear_index(y,xlo,cols); x<=xhi; ++x, ++i) {
                //int i = linear_index(y, x, cols);
                int dist = M*(x-xc)*(x-xc) + ddy;                
                if (dist >= d[i])
                    continue;                
                
                int temp = ((int)img_data[i]-(int)centers[k].color);
                dist += temp*temp;

                if (dist < d[i]) {
                    d[i] = (ushort)dist;
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
    for (int k=0; k<K; ++k) {
        if (is_cluster_invalid(centers[k])) { continue; } 
        int xc = centers[k].x;
        int yc = centers[k].y;   

        int xlo = max(0, xc-search_region_x);
        int xhi = min(cols-1, xc+search_region_x);
        int ylo = max(0, yc-search_region_y);
        int yhi = min(rows-1, yc+search_region_y);

        int nums=0;
        int color=0;
        int sumx=0;
        int sumy=0;
        int x,y,i;
        for (y=ylo; y<=yhi; ++y) {
            for (x=xlo, i=linear_index(y,xlo,cols); x<=xhi; ++x, ++i) {
                if (assignments[i]==k) { 
                    color += (int)img_data[i];
                    sumx += x;
                    sumy += y;
                    nums++;
                }
            }
        }
        if (nums>0) {
            centers[k].x = round( float(sumx) / float(nums) ); 
            centers[k].y = round( float(sumy) / float(nums) );
            centers[k].color = round( float(color) / float(nums) );
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

vector<int> tSlic::get_assignments() { 
    return vector<int>( assignments.begin(), assignments.end() );
}

void tSlic::get_superpixel_pixcount() {
    pixcount = vector<int>(Kmax, 0);
    for (size_t i = 0; i < assignments.size(); ++i) { 
        // note: this should never really happen. that this happens is a symptom
        // of something worse.
        if (assignments[i]>=Kmax) { continue; }
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
            centers[k].color = 0;
            centers[K].color = 0;

            int color1 = 0;
            int color2 = 0;

            vector<float> eigvals = eigvals_2x2(sigma[0], sigma[1], sigma[2], sigma[3]);
            vector<float> eigvec = eigvec_2x2(sigma[0], sigma[1], sigma[2], sigma[3], eigvals[0]);
            
            // update the assignments
            for (int yy = bds[2]; yy <= bds[3]; ++yy) { 
                for (int xx = bds[0]; xx <= bds[1]; ++xx) { 
                    int ids = linear_index(yy, xx, cols);
                    if (assignments[ids] != k) { continue; }
                    int val = (xx-xc_x)*eigvec[0] + (yy-xc_y)*eigvec[1];                    
                    if (val > 0) {
                        set_id = k;
                        n1++;
                        color1 += img_data[ids];
                    } else { 
                        set_id = K;
                        n2++;
                        color2 += img_data[ids];
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
            centers[k].color = ((float)color1/(float)max(n1, 1));
            centers[K].color = ((float)color2/(float)max(n2, 1));
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

        // first check if the centroid (or its 4 nearest neighbors belong to 
        // this superpixel. if yes, great -- we will start at this location.
        // otherwise we need to find a point that belongs to this superpixel
        // while being close to the center (this would be necessary if the 
        // superpixel is U-shaped)
        if ( assignments[ linear_index(yc, xc, cols) ]==k ) { 
            // yay!
            origin_x = xc;
            origin_y = yc;
            nums = 1;
        } else {
            int y,x,i;
            int dist_to_origin = INT_MAX;
            for (y=ylo; y<=yhi; ++y) {
                int ddy = (y-yc)*(y-yc);
                if (ddy > dist_to_origin) { continue; }
                for (x=xlo, i = linear_index(y,xlo,cols); x<=xhi; ++x, ++i) {
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
        int y,x,i;
        for (y = ylo; y<=yhi; ++y) { 
            memset(&visited[0] + linear_index(y,xlo,cols), 0, sizeof(uchar)*(xhi-xlo+1) );
            for (x=xlo, i = linear_index(y,xlo,cols); x<=xhi; ++x, ++i) {
                //visited[i] = false;
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
        //try_fill_iterative(linear_index(origin_y, origin_x, cols), xlosp, xhisp, ylosp, yhisp);
        try_fill(origin_x, origin_y, 
                linear_index(origin_y, origin_x, cols), xlosp, xhisp, ylosp, yhisp);
        
        int num_left = 0;
        for (y=ylosp; y<=yhisp; ++y) {
            for (x=xlosp, i=linear_index(y,xlosp,cols); x<=xhisp; ++x,++i) {
                if (mask[i]) {
                    num_left += 1;
                    break;
                }
            }
            if (num_left > 0) { break; }
        }

        if (num_left > 0) {
            // assign the remaining pixels to neighboring clusters
            reassign_neighbors(k, xlosp, xhisp, ylosp, yhisp);
        }
    }
#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
}


void tSlic::reassign_neighbors(int cluster, int xlo, int xhi, int ylo, int yhi) {
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

void tSlic::try_fill_iterative(int i, int xlo, int xhi, int ylo, int yhi) {
    int x,y;
    deque<int> s;
    s.push_back( i );
    // strict inequalities here, shouldnt go out-of-bounds.
    while (!s.empty()) {
        i = s.front(); s.pop_front();
        visited[i] = true;
        if (mask[i]) {
            mask[i] = false;
            x = getcol(i, cols);
            y = getrow(i, cols);
            // go left
            int i_lef = linear_index(y, x-1, cols);
            if ((x > xlo) && (visited[i_lef] == false))
                s.push_back( linear_index(y, x-1, cols) );
            // go right
            int i_rig = linear_index(y, x+1, cols);
            if ((x < xhi) && (visited[i_rig] == false))
                s.push_back( linear_index(y, x+1, cols) );
            // go bottom
            int i_bot = linear_index(y+1, x, cols);
            if ((y < yhi) && (visited[i_bot] == false))
                s.push_back( linear_index(y+1, x, cols) );
            // go top
            int i_top = linear_index(y-1, x, cols);
            if ((y > ylo) && (visited[i_top] == false))
                s.push_back( linear_index(y-1, x, cols) );
        }
    }
}

void tSlic::try_fill(int x, int y, int i, int xlo, int xhi, int ylo, int yhi) {

    visited[i] = true;
    bool mask_val = mask[i];
    mask[i] = false;
    // strict inequalities here, shouldnt go out-of-bounds.
    if (mask_val) {
        // go left
        int i_lef = linear_index(y, x-1, cols);
        if ((x > xlo) && (visited[i_lef] == false))
            try_fill(x-1, y, i_lef, xlo, xhi, ylo, yhi);
        // go right
        int i_rig = linear_index(y, x+1, cols);
        if ((x < xhi) && (visited[i_rig] == false))
            try_fill(x+1, y, i_rig, xlo, xhi, ylo, yhi);
        // go bottom
        int i_bot = linear_index(y+1, x, cols);
        if ((y < yhi) && (visited[i_bot] == false))
            try_fill(x, y+1, i_bot, xlo, xhi, ylo, yhi);
        // go top
        int i_top = linear_index(y-1, x, cols);
        if ((y > ylo) && (visited[i_top] == false))
            try_fill(x, y-1, i_top, xlo, xhi, ylo, yhi);
    }
}

void tSlic::remove_empty_superpixels() {
#ifdef SLIC_DEBUG
unsigned long t1 = now_us();
#endif
    // clean-up routine.
    get_superpixel_pixcount();

    int numlost = 0;
    vector<tslic_cluster> temp_centers = vector<tslic_cluster>();
    temp_centers.reserve(K);
    for (int i = 0; i < K; ++i) {
        if (is_cluster_invalid(centers[i]) || (pixcount[i]==0)) {
            numlost++;
        } else {
            temp_centers.push_back( centers[i] );
        }
    }
    // lucky case.
    if (numlost == 0) { return; }
   
    // copy the 'good' clusters 
    for (size_t i = 0; i < temp_centers.size(); ++i) { 
        centers[i] = temp_centers[i];
    }
    // invalidate the rest
    for (size_t i = temp_centers.size(); i < centers.size(); ++i) { 
        make_cluster_invalid( centers[i] );
    }

    K = K - numlost;
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
        if (!is_in_range(x,y,0)) {
            make_cluster_invalid(centers[k]);
        } else {
            centers[k].color = img[ linear_index(y, x, cols) ];
        }
    }
}

void tSlic::check_parameters(tslic_params& p) {
    assert( (p.rows>0) && (p.cols>0) 
            && "tslic parameters are not set correctly.");
    assert( (p.chan==1) && "tslic parameters are not set correctly.");
    assert( (p.K>0) && (p.M>0) && (p.num_iters>0)
            && "tslic parameters are not set correctly.");
    assert( (p.min_area_frac>0) && (p.max_area_frac>0) 
            && "tslic parameters are not set correctly.");
}


bool inline tSlic::is_cluster_invalid(tslic_cluster& c) { 
    return (c.id < 0);
}

void inline tSlic::make_cluster_invalid(tslic_cluster& c) {
    c.x = -1;
    c.y = -1;
    c.color = 0;
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
        chan = 1;
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
