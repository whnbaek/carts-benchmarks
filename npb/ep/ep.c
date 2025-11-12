/* NPB-like EP (Embarrassingly Parallel) kernel (single-file, OpenMP) */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M
#define M 24 /* 2^M samples */
#endif

static double rng_state(int k) { /* simple LCG for determinism */
  unsigned int x = 123456789u + 1103515245u * (unsigned)k;
  x = 1103515245u * x + 12345u;
  return (double)(x & 0x7fffffff) / 2147483648.0;
}

int main(void) {
  const long N = 1L << M;
  long inside = 0;

#pragma omp parallel for reduction(+ : inside) schedule(static)
  for (long i = 0; i < N; i++) {
    double x = rng_state((int)(2 * i));
    double y = rng_state((int)(2 * i + 1));
    if (x * x + y * y <= 1.0)
      inside += 1;
  }

  double pi = 4.0 * ((double)inside / (double)N);
  printf("pi=%.9f N=%ld\n", pi, N);
  return 0;
}
