#include "segmentation_utils.h"


bcv_sparse_op<int> create_difference_op(vector<SlicNode<float> >& graph) {
    int nnodes = graph.size();
    int nedges = spgraph_get_num_edges(graph);
    vector<int> p1 = vector<int>();
    vector<int> p2 = vector<int>();
    p1.reserve(nedges);
    p2.reserve(nedges);
    for (size_t i = 0; i < graph.size(); ++i) { 
        for (size_t j = 0; j < graph[i].neighbors.size(); ++j) { 
            p1.push_back(i);
            p2.push_back( graph[i].neighbors[j] );
        }
    }

    return bcv_create_sparse_diff_op_from_data(p1, p2, nnodes);
}

vector<float> compute_unary_potential(const vector<SlicNode<float> >& graph, GMM& fg, GMM& bg) {
    size_t n = graph.size();
    vector<float> unary = vector<float>(n);

    vector<float> val = vector<float>(3);
    for (size_t i = 0; i < n; ++i) {
    	memcpy(&val[0], graph[i].rgb, sizeof(float)*3);
        float p_fg = fg.evaluatePointLikelihood(val);
        float p_bg = bg.evaluatePointLikelihood(val);
        unary[i] = log(p_bg) - log(p_fg); // note the sign!!
    }
    return unary;
}

//! Returns a vector of pairwise weights for all neighboring nodes.
//! The weights are of the form w(x,y) = exp( - beta* ||I(x)-I(y)||^2 )
//! (without any normalization)
vector<float> compute_pairwise_potential(const vector<SlicNode<float> >& graph, 
                                                    float beta, int nedges) { 
    size_t n = graph.size();
    vector<float> out = vector<float>();
    out.reserve(nedges);
    for (size_t i = 0; i < n; ++i) { 
        float d1 = graph[i].rgb[0];
        float d2 = graph[i].rgb[1];
        float d3 = graph[i].rgb[2];

        for (size_t j = 0; j < graph[i].neighbors.size(); ++j) { 
            int ii = graph[i].neighbors[j];
            float e1 = graph[ii].rgb[0];
            float e2 = graph[ii].rgb[1];
            float e3 = graph[ii].rgb[2];
            
            float d = (d1-e1)*(d1-e1) + (d2-e2)*(d2-e2) + (d3-e3)*(d3-e3);
            out.push_back( exp(-beta*d) );
        }
    }
    return out;
}




void learn_appearance_gmm(GMM& fg, GMM& bg, int K, int num_iters , const sss_model& obj) {
    // learn GMMs.
    fg = GMM(K, 1e-4f);
    bg = GMM(K, 1e-4f);
    fg.initParameters();
    bg.initParameters();
    bg.initImageRange(obj.data_bg);
    fg.initImageRange(obj.data_fg);

    for (int i = 0; i < num_iters; ++i) {
        fg.getClusterAssignments(obj.data_fg);
        fg.updateParameters(obj.data_fg);
        bg.getClusterAssignments(obj.data_bg);
        bg.updateParameters(obj.data_bg);
    }
    //fg.print_mixture_weights();
    //bg.print_mixture_weights();
    //fg.print_means();
    //bg.print_means();
    //fg.print_covariances();
    //bg.print_covariances();
}

void learn_appearance_gmm_kmeans(GMM& fg, GMM& bg, int K, int num_iters, const sss_model& obj) {

    bcv_kmeans km;
    vector<int> assignments;
    // foreground
    km = bcv_kmeans(obj.data_fg, obj.num_fg, 3, K, num_iters);
    km.get_assignments(assignments);
    // using kmeans result estimate GMM parameters    
    set_gmm_parameters(fg, obj.data_fg, assignments, K);

    // background
    km = bcv_kmeans(obj.data_bg, obj.num_bg, 3, K, num_iters);
    km.get_assignments(assignments);
    // using kmeans result estimate GMM parameters 
    set_gmm_parameters(bg, obj.data_bg, assignments, K);

}


void set_gmm_parameters(GMM& gmm, const vector<float>& data, vector<int>& assignments, int K) {
    gmm = GMM(K);
    gmm.clusterProb = vector<float>(K, 0.0);
    for (int k = 0; k < K; ++k) {
    	gmm.clusterParam[k].mu = vector<float>(3, 0.0);
    	gmm.clusterParam[k].cov = vector<float>(9, 0.0);
    	gmm.clusterParam[k].covinv = vector<float>(9, 0.0);
    	gmm.clusterParam[k].covdet = 1;
    }
    for (int i = 0; i < assignments.size(); ++i) {
    	gmm.clusterProb[ assignments[i] ]++;
    	for (int u = 0; u < 3; ++u ){
    		gmm.clusterParam[ assignments[i] ].mu[u] += data[3*i+u];
    		gmm.clusterParam[ assignments[i] ].cov[3*u+u] += data[3*i+u]*data[3*i+u];
    	}
    }
    for (int k = 0; k < K; ++k) {
    	float m = max(1.0f, gmm.clusterProb[k]); // this is the number of points in cluster.
    	for (int u = 0; u < 3; ++u ){
    		gmm.clusterParam[ k ].mu[u] /= float(m);
    		gmm.clusterParam[ k ].cov[3*u+u] /= float(m);
    	}
    	gmm.clusterProb[k] /= float(assignments.size());
    }

    for (int k = 0; k < K; ++k) {
    	for (int u = 0; u < 3; ++u ){
    		gmm.clusterParam[k].cov[3*u+u] -=
    				gmm.clusterParam[k].mu[u]*gmm.clusterParam[k].mu[u];
    		gmm.clusterParam[k].cov[3*u+u] = max(1e-3f,
    				gmm.clusterParam[k].cov[3*u+u] );
    	}
    }
    //
    for (int k = 0; k < K; ++k) {
    	gmm.clusterParam[k].covinv[0] = 1/gmm.clusterParam[k].cov[0];
    	gmm.clusterParam[k].covinv[4] = 1/gmm.clusterParam[k].cov[4];
    	gmm.clusterParam[k].covinv[8] = 1/gmm.clusterParam[k].cov[8];

    	gmm.clusterParam[k].covdet =
    			gmm.clusterParam[k].cov[0]*
    			gmm.clusterParam[k].cov[4]*
    			gmm.clusterParam[k].cov[8];
    }
}


void update_bgfg_data(sss_model& obj, 
    const vector<uchar>& mask_vec, const vector<SlicNode<float> >& g) {
    int n_fg = accumulate(mask_vec.begin(), mask_vec.end(), 0);
    int n_bg = mask_vec.size()-n_fg;
    if (obj.data_fg.size()+3*n_fg > obj.data_fg.capacity()) {
        obj.data_fg.reserve( obj.data_fg.size() + 3*n_fg );
    }
    if (obj.data_bg.size()+3*n_bg > obj.data_bg.capacity()) {
        obj.data_bg.reserve( obj.data_bg.size() + 3*n_bg );
    }    
    // split the data
    for (size_t i = 0; i < mask_vec.size(); ++i) {
        if (mask_vec[i]) { 
            for (int k = 0; k < 3; ++k) { obj.data_fg.push_back( g[i].rgb[k] ); }
        } else {
            for (int k = 0; k < 3; ++k) { obj.data_bg.push_back( g[i].rgb[k] ); }
        }
    }
    obj.num_fg = obj.data_fg.size()/3;
    obj.num_bg = obj.data_bg.size()/3;

    // at this point, if either exceeds maximum capacity need to reduce model
    // complexity. at worst, this can be done by kmeans clustering.
}


vector<float> compute_temporal_unary_potential(const vector<tslic_cluster>& nodes,
        const vector<uchar>& old_seg, const vector<int>& old_id) {
    // get a vector of ids
    int n = nodes.size();
    vector<float> unary = vector<float>(n);
    
    // create a 'set' for ids that correspond to nodes that are 'on'.
    set<int> s_on;
    set<int> s_off;

    for (size_t i = 0; i < old_seg.size(); ++i) { 
        if (old_seg[i]) {
            s_on.insert( old_id[i] );
        } else {
            s_off.insert( old_id[i] );
        }
    }
    // go through the current indices, and see if they match anything..
    // each search is log(n)
    for (int i = 0; i < n; ++i) { 
        if (s_on.find( nodes[i].id ) != s_on.end()) { // is in
        // this superpixel exists in the previous segmentation with S(i)=1
        // so DECREASE unary to make current segmentation MORE likely.
            unary[i] = -0.5f;
        } 
        if (s_off.find( nodes[i].id ) != s_off.end()) { // is in
        // this superpixel exists in the previous segmentation with S(i)=0
        // so INCREASE unary to make current segmentation LESS likely.
            unary[i] = +0.5f;
        }
    }
    return unary;
}


void tvseg_iterative_gmm_estimation(GMM& fg_gmm, GMM& bg_gmm,
				sss_model& obj_model, int gmm_num_clusters, int gmm_num_iters, 
				int num_reestimate_iters, int num_segment_iters,
				float beta, float lambda) {

    int sum_og = accumulate(obj_model.supermask.begin(), obj_model.supermask.end(), 0);

    // prepare some parameters (these will not change from iteration to iteration)
	int nedges = spgraph_get_num_edges(obj_model.supergraph);
	tvsegmentbinary_params p;
	p.D = create_difference_op(obj_model.supergraph);
	p.nnodes = obj_model.supergraph.size();
    p.nedges = nedges;
    p.max_iters = num_segment_iters;
    //--------------------------------------------------------------------------
    // set FG/BG using initial estimate
    obj_model.data_fg.clear();
    obj_model.data_bg.clear();
	update_bgfg_data(obj_model, obj_model.supermask, obj_model.supergraph);
    // learn appearance model.
    learn_appearance_gmm_kmeans(fg_gmm, bg_gmm, 
                                gmm_num_clusters, gmm_num_iters, obj_model);
    
    for (int iters = 0; iters < num_reestimate_iters; ++iters) { 

    	// ---------------------------------------------------------------------
    	// 		perform segmentation with the current estimate of GMMs.
    	// ---------------------------------------------------------------------
        p.unary = compute_unary_potential( obj_model.supergraph, fg_gmm, bg_gmm );
        // points that are outside bounding box should be set to large values
        // so that it's expensive to set them to foreground
        for (size_t i = 0; i < p.unary.size(); ++i) {
            p.unary[i] += 1000*(obj_model.supermask[i]==0);
        }
        // compute TV.
        p.weights = compute_pairwise_potential(obj_model.supergraph,
        						beta, nedges);
        transform(p.weights.begin(), p.weights.end(), p.weights.begin(),
                					bind1st(multiplies<float>(), lambda));
        //----------------------------------------------------------------------
    	tvsegmentbinary tvs = tvsegmentbinary(&p);
    	vector<uchar> segvec = tvs.get_segmentation();

        // ----------------------------------------------------------------
        int sum_now = accumulate(segvec.begin(), segvec.end(), 0);
        if ((sum_now == 0) || (sum_now == segvec.size())) { 
        	// this is trouble. this means that everything looks like FG (or BG)
        	// i.e. we reached some kind of degenerate solution. in this case,
        	// just start over.
        	// (would be even better to throw some kind of failure flag....)
        	segvec = obj_model.supermask;
        }
        // ---------------------------------------------------------------------
        //    reestimate foreground/background GMM model using the segmentation
        // ---------------------------------------------------------------------
    	obj_model.data_fg.clear();
    	obj_model.data_bg.clear();
		update_bgfg_data(obj_model, segvec, obj_model.supergraph);
	    // learn appearance model.
	    learn_appearance_gmm_kmeans(fg_gmm, bg_gmm,
	                                gmm_num_clusters, gmm_num_iters, obj_model);
    }
}
