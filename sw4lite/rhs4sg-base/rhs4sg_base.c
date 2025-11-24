#include <math.h>
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
#ifndef POINTS_PER_DIR
#define POINTS_PER_DIR 4
#endif

#define COMP 3
#define IDX(i, j, k) (((i) * NY + (j)) * NZ + (k))

static void init_array(float *u, float *mu, float *lambda, float *rhs) {
  for (int c = 0; c < COMP; ++c) {
    for (int i = 0; i < NX * NY * NZ; ++i) {
      u[c * NX * NY * NZ + i] =
          0.05f * (float)((i + 17 * c) % 23) - 0.1f * (float)c;
      rhs[c * NX * NY * NZ + i] = 0.0f;
    }
  }
  for (int i = 0; i < NX * NY * NZ; ++i) {
    mu[i] = 3.0f + 0.001f * (float)(i % 11);
    lambda[i] = 2.0f + 0.0015f * (float)(i % 7);
  }
}

static void sw4lite_rhs4sg_base(float *restrict rhs,
                                const float *restrict u,
                                const float *restrict mu,
                                const float *restrict lambda, float h) {
  const float w[5] = {-1.0f / 12.0f, 2.0f / 3.0f, 0.0f, -2.0f / 3.0f,
                      1.0f / 12.0f};
  const float inv_h2 = 1.0f / (h * h);

#pragma omp parallel for collapse(2) schedule(static)
  for (int k = POINTS_PER_DIR; k < NZ - POINTS_PER_DIR; ++k) {
    for (int j = POINTS_PER_DIR; j < NY - POINTS_PER_DIR; ++j) {
      for (int i = POINTS_PER_DIR; i < NX - POINTS_PER_DIR; ++i) {
        const int idx = IDX(i, j, k);
        const float mu_c = mu[idx];
        const float la_c = lambda[idx];

        for (int c = 0; c < COMP; ++c) {
          const float *uc = u + c * NX * NY * NZ;
          float lap = 0.0f;
          for (int offset = -2; offset <= 2; ++offset) {
            lap += w[offset + 2] * (uc[IDX(i + offset, j, k)] +
                                    uc[IDX(i, j + offset, k)] +
                                    uc[IDX(i, j, k + offset)]);
          }

          float div_term = 0.0f;
          if (c == 0) {
            div_term = uc[IDX(i + 1, j, k)] - uc[IDX(i - 1, j, k)];
          } else if (c == 1) {
            div_term = uc[IDX(i, j + 1, k)] - uc[IDX(i, j - 1, k)];
          } else {
            div_term = uc[IDX(i, j, k + 1)] - uc[IDX(i, j, k - 1)];
          }

          rhs[c * NX * NY * NZ + idx] =
              mu_c * lap * inv_h2 + (la_c + mu_c) * div_term * (0.5f / h);
        }
      }
    }
  }
}

static float checksum(const float *rhs) {
  float sum = 0.0f;
  for (int c = 0; c < COMP; ++c) {
    for (int i = 0; i < NX * NY * NZ; ++i) {
      sum += rhs[c * NX * NY * NZ + i];
    }
  }
  return sum;
}

int main(void) {
  float *u = (float *)malloc(sizeof(float) * COMP * NX * NY * NZ);
  float *mu = (float *)malloc(sizeof(float) * NX * NY * NZ);
  float *lambda = (float *)malloc(sizeof(float) * NX * NY * NZ);
  float *rhs = (float *)malloc(sizeof(float) * COMP * NX * NY * NZ);
  if (!u || !mu || !lambda || !rhs) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  init_array(u, mu, lambda, rhs);
  sw4lite_rhs4sg_base(rhs, u, mu, lambda, 1.0f);

  printf("sw4lite_rhs4sg_base checksum=%f\n", checksum(rhs));

  free(u);
  free(mu);
  free(lambda);
  free(rhs);
  return 0;
}
