#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NX
#define NX 40
#endif
#ifndef NY
#define NY 40
#endif
#ifndef NZ
#define NZ 40
#endif
#ifndef DT
#define DT 0.001f
#endif

#define IDX(i, j, k) (((i) * NY + (j)) * NZ + (k))

static void init(float *vx, float *vy, float *vz, float *rho,
                 float *mu, float *lambda, float *sxx, float *syy,
                 float *szz, float *sxy, float *sxz, float *syz) {
  for (int i = 0; i < NX * NY * NZ; ++i) {
    vx[i] = 0.001f * (float)(i % 17);
    vy[i] = 0.0015f * (float)((i * 3) % 19);
    vz[i] = 0.0008f * (float)((i * 5) % 23);
    rho[i] = 2500.0f + (float)(i % 7);
    mu[i] = 30.0f + 0.05f * (float)(i % 11);
    lambda[i] = 20.0f + 0.04f * (float)(i % 13);
    sxx[i] = syy[i] = szz[i] = sxy[i] = sxz[i] = syz[i] = 0.0f;
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

static void specfem3d_update_stress(
    float *restrict sxx, float *restrict syy, float *restrict szz,
    float *restrict sxy, float *restrict sxz, float *restrict syz,
    const float *restrict vx, const float *restrict vy,
    const float *restrict vz, const float *restrict mu,
    const float *restrict lambda) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 2; k < NZ - 2; ++k) {
    for (int j = 2; j < NY - 2; ++j) {
      for (int i = 2; i < NX - 2; ++i) {
        const int idx = IDX(i, j, k);
        const float mu_c = mu[idx];
        const float la_c = lambda[idx];

        const float dvx_dx = derivative(vx, i, j, k, 0);
        const float dvy_dy = derivative(vy, i, j, k, 1);
        const float dvz_dz = derivative(vz, i, j, k, 2);

        const float trace = dvx_dx + dvy_dy + dvz_dz;
        const float two_mu = 2.0f * mu_c;

        sxx[idx] += DT * (two_mu * dvx_dx + la_c * trace);
        syy[idx] += DT * (two_mu * dvy_dy + la_c * trace);
        szz[idx] += DT * (two_mu * dvz_dz + la_c * trace);

        sxy[idx] += DT * mu_c * (derivative(vx, i, j, k, 1) +
                                 derivative(vy, i, j, k, 0));
        sxz[idx] += DT * mu_c * (derivative(vx, i, j, k, 2) +
                                 derivative(vz, i, j, k, 0));
        syz[idx] += DT * mu_c * (derivative(vy, i, j, k, 2) +
                                 derivative(vz, i, j, k, 1));
      }
    }
  }
}

static float checksum(const float *arr, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += arr[i];
  return sum;
}

int main(void) {
  const int n = NX * NY * NZ;
  float *vx = (float *)malloc(sizeof(float) * n);
  float *vy = (float *)malloc(sizeof(float) * n);
  float *vz = (float *)malloc(sizeof(float) * n);
  float *rho = (float *)malloc(sizeof(float) * n);
  float *mu = (float *)malloc(sizeof(float) * n);
  float *lambda = (float *)malloc(sizeof(float) * n);
  float *sxx = (float *)malloc(sizeof(float) * n);
  float *syy = (float *)malloc(sizeof(float) * n);
  float *szz = (float *)malloc(sizeof(float) * n);
  float *sxy = (float *)malloc(sizeof(float) * n);
  float *sxz = (float *)malloc(sizeof(float) * n);
  float *syz = (float *)malloc(sizeof(float) * n);
  if (!vx || !vy || !vz || !rho || !mu || !lambda || !sxx || !syy || !szz ||
      !sxy || !sxz || !syz) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  init(vx, vy, vz, rho, mu, lambda, sxx, syy, szz, sxy, sxz, syz);
  specfem3d_update_stress(sxx, syy, szz, sxy, sxz, syz, vx, vy, vz, mu,
                          lambda);

  printf("specfem3d_stress checksum=%f\n",
         checksum(sxx, n) + checksum(syy, n) + checksum(szz, n) +
             checksum(sxy, n) + checksum(sxz, n) + checksum(syz, n));

  free(vx);
  free(vy);
  free(vz);
  free(rho);
  free(mu);
  free(lambda);
  free(sxx);
  free(syy);
  free(szz);
  free(sxy);
  free(sxz);
  free(syz);
  return 0;
}
