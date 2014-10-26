#ifndef BCV_ALG_H_
#define BCV_ALG_H_

#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include "assert.h"


using namespace std;

//! returns a vector of two eigenvalues, if they are real. largest eigenvalue
//! is always the first one. matrix is [a,b; c d]
vector<float> eigvals_2x2(float a, float b, float c, float d);

//! returns the eigenvector correpsonding to a real eigenvalue lambda.
vector<float> eigvec_2x2(float a, float b, float c, float d, float lambda);

#endif
