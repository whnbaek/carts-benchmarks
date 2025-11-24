#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BATCH
#define BATCH 16
#endif

#ifndef HIDDEN
#define HIDDEN 1024
#endif

#ifndef EPS
#define EPS 1e-5f
#endif

static void init(float *x, float *gamma, float *beta) {
  for (int i = 0; i < BATCH * HIDDEN; ++i) {
    x[i] = ((float)(i % 113) - 50.0f) * 0.03125f;
  }
  for (int h = 0; h < HIDDEN; ++h) {
    gamma[h] = 1.0f;
    beta[h] = 0.0f;
  }
}

static void layernorm_forward(float *x, const float *gamma, const float *beta,
                              int batch, int hidden, float eps) {
#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    float mean = 0.0f;
    float var = 0.0f;
    const int offset = b * hidden;
    for (int h = 0; h < hidden; ++h) {
      mean += x[offset + h];
    }
    mean /= hidden;
    for (int h = 0; h < hidden; ++h) {
      float diff = x[offset + h] - mean;
      var += diff * diff;
    }
    var = var / hidden;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int h = 0; h < hidden; ++h) {
      float norm = (x[offset + h] - mean) * inv_std;
      x[offset + h] = norm * gamma[h] + beta[h];
    }
  }
}

static float checksum(const float *x) {
  float sum = 0.0f;
  for (int i = 0; i < BATCH * HIDDEN; ++i) {
    sum += x[i];
  }
  return sum;
}

int main(void) {
  float *x = (float *)malloc(sizeof(float) * BATCH * HIDDEN);
  float *gamma = (float *)malloc(sizeof(float) * HIDDEN);
  float *beta = (float *)malloc(sizeof(float) * HIDDEN);

  if (!x || !gamma || !beta) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  init(x, gamma, beta);
  layernorm_forward(x, gamma, beta, BATCH, HIDDEN, EPS);

  printf("layernorm checksum=%f\n", checksum(x));

  free(x);
  free(gamma);
  free(beta);
  return 0;
}
