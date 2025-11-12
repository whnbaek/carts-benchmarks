/* 2D 5-point stencil with T timesteps (single-file, simple arrays)
   More realistic than trivial loops; no complex data structures. */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1024
#endif

#ifndef TSTEPS
#define TSTEPS 50
#endif

static inline int id(int i, int j, int ld) { return i * ld + j; }

static void init(float *restrict A) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[id(i, j, N)] = (float)((i ^ j) & 0xFF) * 0.01f;
    }
  }
}

static void stencil_step(const float *restrict in, float *restrict out) {
#pragma omp parallel for schedule(static)
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      const float c = in[id(i, j, N)];
      const float n = in[id(i - 1, j, N)];
      const float s = in[id(i + 1, j, N)];
      const float w = in[id(i, j - 1, N)];
      const float e = in[id(i, j + 1, N)];
      out[id(i, j, N)] = 0.2f * (c + n + s + w + e);
    }
  }
}

static float checksum(const float *restrict A) {
  float s = 0.0f;
  for (int i = 0; i < N * N; i++)
    s += A[i];
  return s;
}

int main(void) {
  float *A = (float *)malloc(sizeof(float) * N * N);
  float *B = (float *)malloc(sizeof(float) * N * N);
  if (!A || !B) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }

  init(A);
  // simple Dirichlet boundaries remain as initialized (not updated)
  for (int t = 0; t < TSTEPS; t++) {
    stencil_step(A, B);
    // swap
    float *tmp = A;
    A = B;
    B = tmp;
  }

  printf("checksum=%.6f\n", checksum(A));
  free(A);
  free(B);
  return 0;
}
