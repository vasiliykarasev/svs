#include "bcv_alg.h"

vector<float> eigvals_2x2(float a, float b, float c, float d) {
    vector<float> lambda = vector<float>(2, 0);
    float B = -(a+d);
    float C = a*d - b*c;
    if (B*B < 4*C) { return lambda; } // crap, eigenvalues are not real.
    
    float Z = sqrt(B*B-4*C);
    float l1 = (-B + Z)/2.0f;
    float l2 = (-B - Z)/2.0f;
    if (l1 > l2) { lambda[0] = l1; lambda[1] = l2; }
    else { lambda[0] = l2; lambda[1] = l1; }
    return lambda;
}

vector<float> eigvec_2x2(float a, float b, float c, float d, float lambda) {
    vector<float> v = vector<float>(2);
    a -= lambda;
    d -= lambda;
    // [a,b;c,d] should now be rank 1.
    v[0] = -b;
    v[1] = +a;
    // check: 
    float err = abs(c*v[0] + d*v[1])/abs(lambda);
    if (err > 1e-3f) {
        printf("A = %f %f %f %f\n", a, b, c, d); 
        printf("v = %f %f\n", v[0], v[1]); 
        printf("this is total crap: %f\n", err);
        assert("fuck" && 0);
    }
    return v;
}

