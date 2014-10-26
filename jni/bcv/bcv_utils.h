#ifndef BCV_UTILS_H_
#define BCV_UTILS_H_
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <fstream>
#include <set>
#include "assert.h"

using namespace std;

// image element access operations
int inline linear_index(int r, int c, int k, int cols, int chan) {
    return k + (c + r*cols)*chan;
}
int inline linear_index(int r, int c, int cols) {
    return c + r*cols;
}
int inline getrow(int i, int cols) {
    return i / cols;
}
int inline getcol(int i, int cols) {
    return i - (i / cols)*cols;
}

// io operations

//! Returns 1 if file exists, 0 otherwise.
inline int file_exists(const char* fname) {
  struct stat buffer;
  return (stat(fname, &buffer) == 0); 
}

vector<string> read_file_lines(const char* fname);

//
vector<int> choose_random_subset(int k, int n);

unsigned long now_us();
double now_ms();


#endif // BCV_UTILS_H_
