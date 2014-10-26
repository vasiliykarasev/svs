//! @file SlicGraph.h
#ifndef SlicGraph_H_
#define SlicGraph_H_

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cmath>
#include <climits>
#include <set>
#include <algorithm>
#include "bcv_utils.h"

using namespace std;

typedef unsigned char uchar;
typedef pair<int,int> intpair;
 
template <typename T>
struct SlicNode {
    int x, y;
    T rgb[3];
    vector<int> neighbors;
    vector<pair<int,int> > pts;
    vector<int> pts_linear;
};

//! Constructs a vector of slic-nodes, with information about the superpixels
//! (i.e. pixels they occupy, their neighbors, their average color, etc).
template <typename T> vector<SlicNode<T> > construct_slic_graph(
        vector<unsigned char>& img, const vector<int>& assignments,
        int rows, int cols, int chan) {
#ifdef SLIC_DEBUG
    unsigned long t1,t2;
    t1 = now_us();
    printf("Slic::construct_graph ");
#endif

    // find the total number of superpixels:
    //int n = assignments.size();
    int K = *max_element(assignments.begin(), assignments.end())+1;
    // construct adjacency matrix:
    vector<set<int> > adj = vector< set<int> >(K+1);
    for (int y=0; y<rows-1; ++y) {
        for (int x=0; x<cols-1; ++x) {
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
    vector<SlicNode<T> > graph = vector<SlicNode<T> >(K);
    vector<vector<float> > mean_xy = vector<vector<float> >(K, vector<float>(2,0.0f));
    vector<vector<float> > mean_val = vector<vector<float> >(K, vector<float>(chan,0.0f));
    vector<int> nums = vector<int>(K, 0);
    for (int y = 0; y < rows; ++y) { 
        for (int x = 0; x < cols; ++x) { 
            int sp = assignments[ linear_index(y, x, cols) ];            
            mean_xy[sp][0] += (float)x;
            mean_xy[sp][1] += (float)y;
            nums[sp]++;
            for (int k = 0; k < chan; ++k) { 
                mean_val[sp][k] += (float)img[linear_index(y, x, k, cols, chan)];    
            }
        }
    }
    for (int k = 0; k < K; ++k) { 
        if (nums[k]==0) { continue; } // actually no superpixels should be empty
        mean_xy[k][0] /= (float)nums[k];
        mean_xy[k][1] /= (float)nums[k];
        mean_val[k][0] /= (float)nums[k];
        mean_val[k][1] /= (float)nums[k];
        mean_val[k][2] /= (float)nums[k];
    }
    // now start building the graph:

    for (int i=0; i<K; ++i) {
        graph[i].neighbors = vector<int>( adj[i].size() );
        int u=0;
        for (set<int>::iterator it=adj[i].begin(); it!=adj[i].end(); ++it) {
            graph[i].neighbors[u] = *(it);
            u++;
        }
        if (chan == 3) {
            graph[i].rgb[0] = (T)mean_val[i][0];
            graph[i].rgb[1] = (T)mean_val[i][1];
            graph[i].rgb[2] = (T)mean_val[i][2];
        } else {
            graph[i].rgb[0] = (T)mean_val[i][0];
            graph[i].rgb[1] = (T)mean_val[i][0];
            graph[i].rgb[2] = (T)mean_val[i][0];
        }
        graph[i].x = mean_xy[i][0];
        graph[i].y = mean_xy[i][1];

        graph[i].pts.reserve( ceil(2*rows*cols/K) );
        graph[i].pts_linear.reserve( ceil(2*rows*cols/K) );
    }
    for (int y = 0; y < rows; ++y) { 
        for (int x = 0; x < cols; ++x) {
            int id = linear_index(y,x,cols);
            int k = assignments[ id ];
            graph[k].pts.push_back( pair<int,int>(x,y) );
            graph[k].pts_linear.push_back( id );
        }
    }

#ifdef SLIC_DEBUG
    t2 = now_us();
    printf("elapsed: %f\n", double(t2-t1)/1000.0 );
#endif
    return graph;
}

//! Returns an adjacency list in two vectors, such that p1(i) is a neighbor of p2(i).
//! @param[in] graph
//! @param[out] p1, p2
template <typename T> void spgraph_create_adjacency_list(vector<int>& p1, 
        vector<int>& p2, const vector<SlicNode<T> >& graph) {
    int nnodes = graph.size();
    p1 = vector<int>();
    p2 = vector<int>();
    p1.reserve(nnodes*8);
    p2.reserve(nnodes*8);
    for (size_t i = 0; i < graph.size(); ++i) { 
        for (size_t j = 0; j < graph[i].neighbors.size(); ++j) { 
            p1.push_back(i);
            p2.push_back( graph[i].neighbors[j] );
        }
    }
}

//! Converts a vector of values on superpixels (sp) into an image (img).
//! @param[in] sp - vector of values defined on superpixel regions
//! @param[in] graph - vector of superpixel nodes/data
//! @param[in] rows, cols - size of the output image
//! @param[out] img - vector representing an image (rows x cols x 1) of the same
//! type as the input vector 'sp'.
template <typename T1, typename T2> vector<T1> spgraph_vec2img(
const vector<SlicNode<T2> >& graph, const vector<T1>& sp, int rows, int cols) {
    size_t nsp = sp.size();
    assert( (nsp == graph.size()) && "Number of superpixels matches graph size.\n");
   
    vector<T1> img = vector<T1>(rows*cols, 0);
    for (size_t i = 0; i < nsp; ++i) { 
        size_t m = graph[i].pts_linear.size();
        for (size_t j = 0; j < m; ++j) { 
            img[ graph[i].pts_linear[j] ] = sp[i];
        }
    }
    return img;
} 

//! Converts an image into a vector of values defined on superpixels.
//! @param[in] img, rows, cols - input image. IT IS ASSUMED THAT NUMBER OF
//! CHANNELS IS ONE!!!
//! @param[in] graph - vector of superpixel nodes/data
//! @param[in] type - type of aggregation (see below)
//! @param[out] sp - vector of values defined on superpixels
//!
//! this transformation is lossy; it asks to convert multiple values into one.
//! there is no standard way to do this; in some situations 'mean' is appropriate
//! while in others, it should be the value that most frequently appears.
//! the method of aggregation is governed by 'type'; in particular:
//! type = 0 ---> mean (values are averaged over the superpixel)
//! type = 1 ---> most frequent value
template <typename T1, typename T2> vector<T1> spgraph_img2vec(
const vector<SlicNode<T2> >& graph, const vector<T1>& img, int rows, int cols, int type) {
    int n = img.size();
    assert( (n/rows/cols == 1) && ("Must be a single channel image") );
    
    int nsp = graph.size();
    vector<T1> sp = vector<T1>( nsp, 0);
    for (int i = 0; i < nsp; ++i) { 
        int m = graph[i].pts_linear.size();
        //---------------------------------------------------------------------
        if (type == 0) { //  perform averaging  
            float s = 0.0f;
            for (int j = 0; j < m; ++j) { 
                s += img[ graph[i].pts_linear[j] ];
            }
            sp[i] = (s/float(m));
        } else if (type == 1) { // take the most frequently appearing value
            // get unique elements
            set<T1> setvals;
            for (int j = 0; j < m; ++j) {
                setvals.insert( img[graph[i].pts_linear[j] ] );
            }
            // count them:
            vector<int> counts = vector<int>( setvals.size() , 0);
            vector<T1>  vals = vector<T1>( setvals.size() , 0);
            
            for (int j = 0; j < m; ++j) { 
                T1 pt = img[ graph[i].pts_linear[j] ];
                int idx = distance( setvals.begin(), setvals.find(pt) );
                counts[idx]++;
                vals[idx] = pt;
            }
            // get most frequent element:
            int max_idx = 0;
            if (m>0) {
                max_idx = distance( counts.begin(), 
                    max_element(counts.begin(), counts.begin() + counts.size() ) );
            }
            sp[i] = vals[max_idx]; 
        } else {
            printf(" spgraph_img2vec NOT IMPLEMENTED. \n");
        }
    }
    return sp;
} 


//! Writes information about the superpixels into an ASCII file
template <typename T> void write_slic_graph(const vector<SlicNode<T> >& graph, const char* fname) {
    FILE* fid;
    fid = fopen(fname,"w");
    if (fid == NULL) { 
        printf("Could not open %s for writing\n", fname);
        return;
    }
    // each line LAB of node, followed by indices of neighbors
    int K = graph.size();
    for (int k=0; k<K; ++k) {
        fprintf(fid, "%d %d %d ", 
                graph[k].rgb[0], graph[k].rgb[1], graph[k].rgb[2]);
        int m = graph[k].neighbors.size();
        for (int u=0; u<m; ++u) {
            fprintf(fid, "%d ", graph[k].neighbors[u]);
        }
        fprintf(fid, "\n"); 
    }
    fclose(fid);
}

//! Writes superpixel center locations, and their neighbors to an ASCII file
template <typename T> void write_slic_graph_geometry(
                        const vector<SlicNode<T> >& graph, const char* fname) {
    FILE* fid;
    fid = fopen(fname,"w");
    if (fid == NULL) { 
        printf("Could not open %s for writing\n", fname);
        return;
    }
    // each line contains (x,y) of the node, followed by (x,y) locations
    // of the neighbors.
    int K = graph.size();
    for (int k=0; k<K; ++k) {
        fprintf(fid, "%d %d ", graph[k].x, graph[k].y);
        int m = graph[k].neighbors.size();
        for (int u=0; u<m; ++u) {
            int uu = graph[k].neighbors[u];
            fprintf(fid, "%d %d ", graph[uu].x, graph[uu].y );
        }
        fprintf(fid, "\n"); 
    }
    fclose(fid);
}

// ! Returns an RGB image with superpixels shown with their average color
template <typename T> vector<unsigned char> slic_average_image(
        const vector<SlicNode<T> >& graph, int rows, int cols, int chan) {
    vector<unsigned char> img = vector<unsigned char>(rows*cols*chan);
    int K = graph.size();
    for (int i = 0; i < K; ++i) {
        int m = graph[i].pts.size();
        for (int j = 0; j < m; ++j) { 
            int x = graph[i].pts[j].first;
            int y = graph[i].pts[j].second;
            if (chan == 3) {
                for (int u=0; u<3; ++u) {
                    int q = linear_index(y, x, u, cols, chan);
                    img[q] = graph[i].rgb[u];
                }
            } else {
                int q = linear_index(y, x, cols);
                img[q] = graph[i].rgb[0];
            }
        }
    }
    return img;
}

template <typename T> void slic_scale_graph_pixels(
                                    vector<SlicNode<T> >& graph, float s) {
    int n = graph.size();
    for (int i = 0; i < n; ++i) { 
        graph[i].rgb[0] *= s;
        graph[i].rgb[1] *= s;
        graph[i].rgb[2] *= s;
    }
}

template <typename T> int spgraph_get_num_edges(const vector<SlicNode<T> >& g) {
    int n = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        n += g[i].neighbors.size();
    }
    return n;
}

template <typename T> void spgraph_vis_edges(vector<uchar>& img, 
                    const vector<SlicNode<T> >& g, const vector<float>& edges, 
                    int rows, int cols, int chan) {
    int k = 0;
    uchar val[3]; memset(val,0,sizeof(uchar)*3);
    for (size_t i = 0; i < g.size(); ++i) {
        int x0 = g[i].x;
        int y0 = g[i].y;

        for (size_t j = 0; j < g[i].neighbors.size(); ++j) {
            int ii = g[i].neighbors[j];
            int x1 = g[ii].x;
            int y1 = g[ii].y;

            int x0_ = x0;
            int y0_ = y0;            
            
            float e = edges[k];
            val[0] = 255*e; // convert edge value to 'color'.
            { // draw a line going from (x0_,y0_) to (x1,y1)
                int dx = abs(x1-x0_), sx = x0_ < x1 ? 1 : -1;
                int dy = abs(y1-y0_), sy = y0_ < y1 ? 1 : -1; 
                int err = (dx > dy ? dx : -dy)/2, e2;
             
                for(;;) {
                    for (int ch = 0; ch < chan; ++ch) {
                        img[ linear_index(y0_,x0_,ch,cols,chan) ] = val[ch];
                    }
                    if (x0_ == x1 && y0_ == y1) { break; }
                    e2 = err;
                    if (e2 >-dx) { err -= dy; x0_ += sx; }
                    if (e2 < dy) { err += dx; y0_ += sy; }
                }
            }
            //
            k++;
        }
    }
}

//! Concatenates two superpixel graphs (e.g. from different images)
//! Resulting graph's pts (and pts_linear) are meaningless and thus cleared.
template <typename T> void spgraph_append(
                vector<SlicNode<T> >& g1, const vector<SlicNode<T> >& g2) { 
    int n1 = g1.size();
    int n2 = g2.size();

    g1.insert( g1.end(), g2.begin(), g2.end() );
    // correct neighbors
    for (int i = n1; i < n1+n2; ++i) { 
        int m = g1[i].neighbors.size();
        for (int j = 0; j < m; ++j) { 
            g1[i].neighbors[j] += n1;
        }
    }
    // invalidate points (since concatenation of graphs will typically be from
    // different images)
    for (int i = 0; i < g1.size(); ++i) { 
        g1[i].pts.clear();
        g1[i].pts_linear.clear();
    }
}

//! Given pairs of indices into graph 'g', given by 'p1' and 'p2', adds
//! corresponding neighborhood structure into the graph.
template <typename T> void spgraph_add_neighbors( vector<SlicNode<T> >& g,
                            const vector<int>& p1, const vector<int>& p2) {
    assert( (p1.size() == p2.size()) && "Sizes must be equal.");
    size_t n = p1.size();
    for (size_t i = 0; i < n; ++i) { 
        int n1 = p1[i];
        int n2 = p2[i];
        printf("n1,n2: %d %d, siez: %d\n", n1,n2,g.size() );
        g[n1].neighbors.push_back( n2 );
        g[n2].neighbors.push_back( n1 );
    } 
}

//! Given two sets of id (int previous and current frame), 
//! returns a pair of vectors, where p1,p2 = (i,j) satisfies id_prev(i) = id_cur(j).
//! In other words, pairs are indices into id_prev and id_cur.
//! This function is SLOW -- O(n^2)
template <typename T> void spgraph_get_id_pairs(vector<int>& p1, vector<int>& p2, 
                                    const vector<int>& id_prev, 
                                    const vector<int>& id_cur) {
    int n = min( id_prev.size(), id_cur.size() );
    
    p1.clear(); 
    p2.clear(); 
    p1.reserve( n );
    p2.reserve( n );

    set<int> s;
    for (size_t i = 0; i < id_prev.size(); ++i) { 
        for (size_t j = 0; j < id_cur.size(); ++j) { 
            if ((id_prev[i] == id_cur[j]) && (s.find(id_prev[i]) == s.end())) {
                s.insert( id_prev[i] );
                p1.push_back(i);
                p2.push_back(j);
            }
        }
    }
}

#endif // SlicGraph_H_
