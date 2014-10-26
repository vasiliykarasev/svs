#include "bcv_sparse_op.h"

bcv_sparse_op<int> bcv_create_sparse_diff_op_from_data(const vector<int>& p1, 
        const vector<int>& p2, int n_opcols) {
    bcv_sparse_op<int> D;
    int m = p1.size();
    D.R = vector<int>();
    D.C = vector<int>();
    D.val = vector<int>();
    D.R.reserve(2*m);
    D.C.reserve(2*m);
    D.val.reserve(2*m);
    D.nrows = m;
    D.ncols = n_opcols;   

    // each row is of the form: (Dx)_i = x[ p2[i] ] - x[ p1[i] ]
    for (int i = 0; i < m; ++i) {  
        D.R.push_back( i );
        D.C.push_back( p1[i] );
        D.val.push_back( -1 );

        D.R.push_back( i );
        D.C.push_back( p2[i] );
        D.val.push_back( +1 );
    }
    return D;
}
