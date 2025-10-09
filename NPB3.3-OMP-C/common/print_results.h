#ifndef PRINT_RESULTS_H
#define PRINT_RESULTS_H

#ifndef NPB_NO_STDIO
#include <stdio.h>
#endif

void c_print_results(char *name, char class, int n1, int n2, int n3, int niter,
                     double t, double mops, char *optype, int passed_verification,
                     char *npbversion, char *compiletime, char *cc, char *clink, char *c_lib,
                     char *c_inc, char *cflags, char *clinkflags, char *rand);

#endif
