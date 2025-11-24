#include <stdio.h>
#include <stdlib.h>

#ifndef NX
#define NX 48
#endif
#ifndef NY
#define NY 48
#endif
#ifndef NZ
#define NZ 48
#endif
#ifndef DT
#define DT 0.0005f
#endif

#define COMP 3
#define IDX(i, j, k) (((i) * NY + (j)) * NZ + (k))

static void init(float *vx, float *vy, float *vz, float *rho, float *sxx,
                 float *syy, float *szz, float *sxy, float *sxz, float *syz) {
  for (int i = 0; i < NX * NY * NZ; ++i) {
    vx[i] = 0.0f;
    vy[i] = 0.0f;
    vz[i] = 0.0f;
    rho[i] = 2500.0f + (float)(i % 13);
    sxx[i] = 0.01f * (float)((i * 7) % 19);
    syy[i] = 0.01f * (float)((i * 11) % 23);
    szz[i] = 0.01f * (float)((i * 13) % 17);
    sxy[i] = 0.005f * (float)((i * 5) % 29);
    sxz[i] = 0.004f * (float)((i * 3) % 31);
    syz[i] = 0.006f * (float)((i * 2) % 37);
  }
}

static inline float derivative(const float *arr, int i, int j, int k,
                               int dir) {
  switch (dir) {
  case 0:
    return 0.5f *
           (arr[IDX(i + 1, j, k)] - arr[IDX(i - 1, j, k)]);
  case 1:
    return 0.5f *
           (arr[IDX(i, j + 1, k)] - arr[IDX(i, j - 1, k)]);
  default:
    return 0.5f *
           (arr[IDX(i, j, k + 1)] - arr[IDX(i, j, k - 1)]);
  }
}

static void sw4lite_vel4sg_update(float *restrict vx, float *restrict vy,
                                  float *restrict vz,
                                  const float *restrict rho,
                                  const float *restrict sxx,
                                  const float *restrict syy,
                                  const float *restrict szz,
                                  const float *restrict sxy,
                                  const float *restrict sxz,
                                  const float *restrict syz) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 2; k < NZ - 2; ++k) {
    for (int j = 2; j < NY - 2; ++j) {
      for (int i = 2; i < NX - 2; ++i) {
        const int idx = IDX(i, j, k);
        const float inv_rho = 1.0f / rho[idx];

        const float div_vx = derivative(sxx, i, j, k, 0) +
                             derivative(sxy, i, j, k, 1) +
                             derivative(sxz, i, j, k, 2);
        const float div_vy = derivative(sxy, i, j, k, 0) +
                             derivative(syy, i, j, k, 1) +
                             derivative(syz, i, j, k, 2);
        const float div_vz = derivative(sxz, i, j, k, 0) +
                             derivative(syz, i, j, k, 1) +
                             derivative(szz, i, j, k, 2);

        vx[idx] += DT * inv_rho * div_vx;
        vy[idx] += DT * inv_rho * div_vy;
        vz[idx] += DT * inv_rho * div_vz;
      }
    }
  }
}

static float checksum(const float *vx, const float *vy, const float *vz) {
  float sum = 0.0f;
  for (int i = 0; i < NX * NY * NZ; ++i) {
    sum += vx[i] + vy[i] + vz[i];
  }
  return sum;
}

int main(void) {
  const int n = NX * NY * NZ;
  float *vx = (float *)malloc(sizeof(float) * n);
  float *vy = (float *)malloc(sizeof(float) * n);
  float *vz = (float *)malloc(sizeof(float) * n);
  float *rho = (float *)malloc(sizeof(float) * n);
  float *sxx = (float *)malloc(sizeof(float) * n);
  float *syy = (float *)malloc(sizeof(float) * n);
  float *szz = (float *)malloc(sizeof(float) * n);
  float *sxy = (float *)malloc(sizeof(float) * n);
  float *sxz = (float *)malloc(sizeof(float) * n);
  float *syz = (float *)malloc(sizeof(float) * n);
  if (!vx || !vy || !vz || !rho || !sxx || !syy || !szz || !sxy || !sxz ||
      !syz) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  init(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);
  sw4lite_vel4sg_update(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);

  printf("sw4lite_vel4sg_base checksum=%f\n", checksum(vx, vy, vz));

  free(vx);
  free(vy);
  free(vz);
  free(rho);
  free(sxx);
  free(syy);
  free(szz);
  free(sxy);
  free(sxz);
  free(syz);
  return 0;
}
