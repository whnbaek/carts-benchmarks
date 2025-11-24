#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LENGTH
#define LENGTH 1048576
#endif

#ifndef ALPHA
#define ALPHA 1.2345f
#endif

static void init(float *x, float *y) {
  for (int i = 0; i < LENGTH; ++i) {
    x[i] = (float)(i % 97) * 0.01f;
    y[i] = (float)(i % 71) * 0.02f;
  }
}

static void axpy(float alpha, const float *x, float *y) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < LENGTH; ++i) {
    y[i] = alpha * x[i] + y[i];
  }
}

static float checksum(const float *y) {
  float sum = 0.0f;
  for (int i = 0; i < LENGTH; ++i) {
    sum += y[i];
  }
  return sum;
}

int main(void) {
  float *x = (float *)malloc(sizeof(float) * LENGTH);
  float *y = (float *)malloc(sizeof(float) * LENGTH);
  if (!x || !y) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  init(x, y);
  axpy(ALPHA, x, y);

  printf("axpy checksum=%f\n", checksum(y));

  free(x);
  free(y);
  return 0;
}
