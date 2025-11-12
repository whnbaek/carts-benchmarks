/* STREAM Triad-like kernel (single-file, OpenMP parallel for)
   Isolated and simple: flat arrays, inlinable helpers, no complex data structures. */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1048576 /* 1M elements */
#endif

#ifndef REPS
#define REPS 5
#endif

static void initialize(float *restrict a, float *restrict b, float *restrict c, float scalar) {
  for (int i = 0; i < N; i++) {
    a[i] = 0.0f;
    b[i] = (float)i * 0.5f;
    c[i] = (float)i * 0.25f;
  }
  (void)scalar;
}

static void triad(float *restrict a, const float *restrict b, const float *restrict c, float scalar) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    a[i] = b[i] + scalar * c[i];
  }
}

static float checksum(const float *restrict a) {
  float s = 0.0f;
  for (int i = 0; i < N; i++) s += a[i];
  return s;
}

int main(void) {
  float *a = (float *)malloc(sizeof(float) * N);
  float *b = (float *)malloc(sizeof(float) * N);
  float *c = (float *)malloc(sizeof(float) * N);
  if (!a || !b || !c) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }

  const float scalar = 3.0f;
  initialize(a, b, c, scalar);

  for (int r = 0; r < REPS; r++) {
    triad(a, b, c, scalar);
  }

  printf("checksum=%.6f\n", checksum(a));
  free(a);
  free(b);
  free(c);
  return 0;
}


