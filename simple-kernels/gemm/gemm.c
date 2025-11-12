/* Naive GEMM C = alpha*A*B + beta*C (single-file)
   Isolated: flat arrays, inlinable helpers, OpenMP parallel for collapse(2). */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 256
#endif

static inline int idx(int r, int c, int ld) { return r * ld + c; }

static void init(float *restrict A, float *restrict B, float *restrict C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[idx(i,j,N)] = (float)((i + j) % 13) * 0.01f;
      B[idx(i,j,N)] = (float)((i - j + N) % 17) * 0.02f;
      C[idx(i,j,N)] = 0.0f;
    }
  }
}

static void gemm(float *restrict C, const float *restrict A, const float *restrict B, float alpha, float beta) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += A[idx(i,k,N)] * B[idx(k,j,N)];
      }
      C[idx(i,j,N)] = alpha * sum + beta * C[idx(i,j,N)];
    }
  }
}

static float checksum(const float *restrict C) {
  float s = 0.0f;
  for (int i = 0; i < N*N; i++) s += C[i];
  return s;
}

int main(void) {
  float *A = (float *)malloc(sizeof(float) * N * N);
  float *B = (float *)malloc(sizeof(float) * N * N);
  float *C = (float *)malloc(sizeof(float) * N * N);
  if (!A || !B || !C) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  init(A, B, C);
  gemm(C, A, B, 1.0f, 0.0f);
  printf("checksum=%.6f\n", checksum(C));
  free(A);
  free(B);
  free(C);
  return 0;
}


