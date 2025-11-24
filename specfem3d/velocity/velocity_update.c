#include <omp.h>
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
#define DT 0.001f
#endif

#define IDX(i, j, k) (((i) * NY + (j)) * NZ + (k))

static void init(float *vx, float *vy, float *vz, float *rho, float *sxx,
                 float *syy, float *szz, float *sxy, float *sxz, float *syz) {
  for (int i = 0; i < NX * NY * NZ; ++i) {
    vx[i] = 0.0f;
    vy[i] = 0.0f;
    vz[i] = 0.0f;
    rho[i] = 2300.0f + (float)(i % 11);
    sxx[i] = 0.02f * (float)((i * 2) % 17);
    syy[i] = 0.02f * (float)((i * 3) % 19);
    szz[i] = 0.02f * (float)((i * 5) % 23);
    sxy[i] = 0.01f * (float)((i * 7) % 13);
    sxz[i] = 0.01f * (float)((i * 11) % 29);
    syz[i] = 0.01f * (float)((i * 13) % 31);
  }
}

static inline float diff(const float *arr, int i, int j, int k, int dir) {
  switch (dir) {
  case 0:
    return arr[IDX(i + 1, j, k)] - arr[IDX(i, j, k)];
  case 1:
    return arr[IDX(i, j + 1, k)] - arr[IDX(i, j, k)];
  default:
    return arr[IDX(i, j, k + 1)] - arr[IDX(i, j, k)];
  }
}

static void specfem_velocity_update(
    float *restrict vx, float *restrict vy, float *restrict vz,
    const float *restrict rho, const float *restrict sxx,
    const float *restrict syy, const float *restrict szz,
    const float *restrict sxy, const float *restrict sxz,
    const float *restrict syz) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 1; k < NZ - 1; ++k) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int i = 1; i < NX - 1; ++i) {
        const int idx = IDX(i, j, k);
        const float inv_rho = 1.0f / rho[idx];

        const float dvx =
            diff(sxx, i, j, k, 0) + diff(sxy, i, j, k, 1) +
            diff(sxz, i, j, k, 2);
        const float dvy =
            diff(sxy, i, j, k, 0) + diff(syy, i, j, k, 1) +
            diff(syz, i, j, k, 2);
        const float dvz =
            diff(sxz, i, j, k, 0) + diff(syz, i, j, k, 1) +
            diff(szz, i, j, k, 2);

        vx[idx] += DT * inv_rho * dvx;
        vy[idx] += DT * inv_rho * dvy;
        vz[idx] += DT * inv_rho * dvz;
      }
    }
  }
}

static float checksum(const float *vx, const float *vy, const float *vz) {
  float s = 0.0f;
  for (int i = 0; i < NX * NY * NZ; ++i)
    s += vx[i] + vy[i] + vz[i];
  return s;
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
  specfem_velocity_update(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);

  printf("specfem3d_velocity checksum=%f\n", checksum(vx, vy, vz));

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
