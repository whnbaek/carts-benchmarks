/* PolyBench-like GEMM (single-file, OpenMP) */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NI
#define NI 512
#endif
#ifndef NJ
#define NJ 512
#endif
#ifndef NK
#define NK 512
#endif

static inline int id(int r, int c, int ld) { return r * ld + c; }

static void init(float *restrict A, float *restrict B, float *restrict C) {
  for (int i = 0; i < NI; i++)
    for (int k = 0; k < NK; k++)
      A[id(i,k,NK)] = (float)((i + k) % 13) * 0.01f;
  for (int k = 0; k < NK; k++)
    for (int j = 0; j < NJ; j++)
      B[id(k,j,NJ)] = (float)((k * 3 + j) % 17) * 0.02f;
  for (int i = 0; i < NI; i++)
    for (int j = 0; j < NJ; j++)
      C[id(i,j,NJ)] = 0.0f;
}

static void gemm(float *restrict C, const float *restrict A, const float *restrict B, float alpha, float beta) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
      float sum = 0.0f;
      for (int k = 0; k < NK; k++) sum += A[id(i,k,NK)] * B[id(k,j,NJ)];
      C[id(i,j,NJ)] = alpha * sum + beta * C[id(i,j,NJ)];
    }
  }
}

static float checksum(const float *restrict C) {
  float s = 0.0f;
  for (int i = 0; i < NI * NJ; i++) s += C[i];
  return s;
}

int main(void) {
  float *A = (float *)malloc(sizeof(float) * NI * NK);
  float *B = (float *)malloc(sizeof(float) * NK * NJ);
  float *C = (float *)malloc(sizeof(float) * NI * NJ);
  if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
  init(A, B, C);
  gemm(C, A, B, 1.0f, 0.0f);
  printf("checksum=%.6f\n", checksum(C));
  free(A); free(B); free(C);
  return 0;
}


