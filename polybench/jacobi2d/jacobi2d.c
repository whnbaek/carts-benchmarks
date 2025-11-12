/* PolyBench-like 2D Jacobi (single-file, OpenMP) */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1024
#endif
#ifndef TSTEPS
#define TSTEPS 100
#endif

static inline int id(int i, int j, int ld) { return i * ld + j; }

static void init(float *restrict A, float *restrict B) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[id(i,j,N)] = (float)((i + j) % 256) * 0.001f;
      B[id(i,j,N)] = 0.0f;
    }
  }
}

static void jacobi2d_step(const float *restrict A, float *restrict B) {
#pragma omp parallel for schedule(static)
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      B[id(i,j,N)] = 0.2f * (A[id(i,j,N)] + A[id(i-1,j,N)] + A[id(i+1,j,N)] + A[id(i,j-1,N)] + A[id(i,j+1,N)]);
    }
  }
}

static float checksum(const float *restrict A) {
  float s = 0.0f;
  for (int i = 0; i < N*N; i++) s += A[i];
  return s;
}

int main(void) {
  float *A = (float *)malloc(sizeof(float) * N * N);
  float *B = (float *)malloc(sizeof(float) * N * N);
  if (!A || !B) { fprintf(stderr, "alloc failed\n"); return 1; }
  init(A, B);
  for (int t = 0; t < TSTEPS; t++) {
    jacobi2d_step(A, B);
    float *tmp = A; A = B; B = tmp;
  }
  printf("checksum=%.6f\n", checksum(A));
  free(A); free(B);
  return 0;
}


