#include "../common/main.h"

#ifndef MSIZE
#define MSIZE 1
#endif
#ifndef SMSIZE
#define SMSIZE 1
#endif

int main() {
    struct user_parameters p;
    p.string2display = "sparselu";
    p.niter = 1;
#ifdef MSIZE
    p.matrix_size = 128;
#endif
#ifdef SMSIZE
    p.submatrix_size = 16;
#endif
    p.check = 0;
    p.succeed = 1;
    return 0;
}
