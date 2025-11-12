/*
 * KaStORS Jacobi-For Benchmark
 * Adapted for CARTS compiler - Single file version (Simplified)
 *
 * Original: https://github.com/viroulep/kastors
 * License: GNU LGPL
 * Author: John Burkardt (modified by KaStORS team)
 *
 * Adaptation changes:
 * - Simplified OpenMP structure to avoid nested parallel/for
 * - Changed to parallel for directly (avoiding nested pragma issue)
 * - Self-contained single file for CARTS compilation
 * - Kernel logic unchanged
 */

#include <stdlib.h>

/*
 * Simplified sweep function using direct parallel for loops
 * instead of parallel region with nested for loops
 */
static void sweep(int nx, int ny, double dx, double dy, double *f_, int itold,
                  int itnew, double *u_, double *unew_, int block_size) {
  int i, j, it;
  double (*f)[nx][ny] = (double (*)[nx][ny])f_;
  double (*u)[nx][ny] = (double (*)[nx][ny])u_;
  double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;

  for (it = itold + 1; it <= itnew; it++) {
    /* Save the current estimate */
#pragma omp parallel for private(j)
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        (*u)[i][j] = (*unew)[i][j];
      }
    }

    /* Compute a new estimate */
#pragma omp parallel for private(j)
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
          (*unew)[i][j] = (*f)[i][j];
        } else {
          (*unew)[i][j] =
              0.25 * ((*u)[i - 1][j] + (*u)[i][j + 1] + (*u)[i][j - 1] +
                      (*u)[i + 1][j] + (*f)[i][j] * dx * dy);
        }
      }
    }
  }
}

int main(void) {
  int nx = 100, ny = 100;
  int itold = 0, itnew = 10;
  int block_size = 10;
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  double *f = (double *)malloc(nx * ny * sizeof(double));
  double *u = (double *)malloc(nx * ny * sizeof(double));
  double *unew = (double *)malloc(nx * ny * sizeof(double));

  /* Initialize arrays */
  for (int i = 0; i < nx * ny; i++) {
    f[i] = 0.0;
    u[i] = 0.0;
    unew[i] = 0.0;
  }

  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);

  free(f);
  free(u);
  free(unew);
  return 0;
}
