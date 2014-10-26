//! @file bcv_basic.cpp
#include "bcv_utils.h"

//! returns time in microseconds
unsigned long now_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long ret = tv.tv_usec;
    ret += (tv.tv_sec * 1000*1000);
    return ret;
}
//! returns time in milliseconds
double now_ms() {
    return double(now_us())/1000.0f;
}


//! Returns a randomly chosen subset S, s.t. |S|=k, sampled from {0,...,n-1}
vector<int> choose_random_subset(int k, int n) {
    assert( (k<=n) && "asked for subset smaller than n");
    
    vector<int> s;
    s.reserve(k);
    if (k==n) { 
        for (int i = 0; i < n; ++i) { s[i] = i; }
        return s;
    }
    // TODO: rewrite this with set
    set<int> s_;
    while (s_.size() < k) {
        int idx = rand() % n;
        s_.insert(idx);
    }
    // move to vector
    for (set<int>::iterator it=s_.begin(); it!=s_.end(); ++it) {
        s.push_back( *(it) );
    }
    return s;
}

//! Returns a vector of strings, where each string is a line from a given file.
vector<string> read_file_lines(const char* fname) {
    string l;
    ifstream fp(fname);
    vector<string> lines;
    if (!fp.is_open()) {
        printf("Could not open %s\n", fname);
        return lines;
    }
    while (getline(fp,l)) {
        // trim trailing spaces
        size_t endpos = l.find_last_not_of(" \t");
        if( string::npos != endpos ) {
            l = l.substr( 0, endpos+1 );
        }
        lines.push_back(l);
    }
    fp.close();
    return lines;
}
