#include "../common/main.h"

#ifndef MSIZE
#define MSIZE 1
#endif
#ifndef CUTOFF_SIZE
#define CUTOFF_SIZE 1
#endif
#ifndef CUTOFF_DEPTH
#define CUTOFF_DEPTH 1
#endif

int main() {
    struct user_parameters p;
    p.string2display = "strassen";
    p.niter = 1;
#ifdef MSIZE
    p.matrix_size = 128;
#endif
#ifdef CUTOFF_SIZE
    p.cutoff_size = 32;
#endif
#ifdef CUTOFF_DEPTH
    p.cutoff_depth = 2;
#endif
    p.check = 0;
    p.succeed = 1;
    return 0;
}
