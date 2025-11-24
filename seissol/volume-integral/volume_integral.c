#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef N_ELEMENTS
#define N_ELEMENTS 64
#endif

#ifndef N_BASIS
#define N_BASIS 30
#endif

#ifndef N_QUAD
#define N_QUAD 36
#endif

static void init(float *dofs, float *gradMatrix, float *fluxMatrix) {
  for (int e = 0; e < N_ELEMENTS * N_BASIS; ++e) {
    dofs[e] = 0.01f * (float)((e * 7) % 23);
  }
  for (int q = 0; q < N_QUAD * N_BASIS; ++q) {
    gradMatrix[q] = 0.001f * (float)((q * 13) % 29);
  }
  for (int q = 0; q < N_QUAD * N_BASIS; ++q) {
    fluxMatrix[q] = 0.002f * (float)((q * 17) % 31);
  }
}

static void seissol_volume_integral(float *restrict fluxOut,
                                    const float *restrict dofs,
                                    const float *restrict gradMatrix,
                                    const float *restrict fluxMatrix) {
#pragma omp parallel for schedule(static)
  for (int elem = 0; elem < N_ELEMENTS; ++elem) {
    const float *elemDofs = dofs + elem * N_BASIS;
    float buffer[N_QUAD];
    for (int q = 0; q < N_QUAD; ++q) {
      float val = 0.0f;
      for (int b = 0; b < N_BASIS; ++b) {
        val += gradMatrix[q * N_BASIS + b] * elemDofs[b];
      }
      buffer[q] = val;
    }
    for (int b = 0; b < N_BASIS; ++b) {
      float acc = 0.0f;
      for (int q = 0; q < N_QUAD; ++q) {
        acc += fluxMatrix[q * N_BASIS + b] * buffer[q];
      }
      fluxOut[elem * N_BASIS + b] = acc;
    }
  }
}

static float checksum(const float *fluxOut) {
  float s = 0.0f;
  for (int e = 0; e < N_ELEMENTS * N_BASIS; ++e) {
    s += fluxOut[e];
  }
  return s;
}

int main(void) {
  float *dofs = (float *)malloc(sizeof(float) * N_ELEMENTS * N_BASIS);
  float *gradMatrix = (float *)malloc(sizeof(float) * N_QUAD * N_BASIS);
  float *fluxMatrix = (float *)malloc(sizeof(float) * N_QUAD * N_BASIS);
  float *fluxOut = (float *)malloc(sizeof(float) * N_ELEMENTS * N_BASIS);
  if (!dofs || !gradMatrix || !fluxMatrix || !fluxOut) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  init(dofs, gradMatrix, fluxMatrix);
  seissol_volume_integral(fluxOut, dofs, gradMatrix, fluxMatrix);

  printf("seissol_volume_integral checksum=%f\n", checksum(fluxOut));

  free(dofs);
  free(gradMatrix);
  free(fluxMatrix);
  free(fluxOut);
  return 0;
}
