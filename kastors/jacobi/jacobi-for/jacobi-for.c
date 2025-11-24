/*
 * KaStORS Jacobi-For Benchmark
 * Adapted for CARTS compiler - Array-of-arrays version
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
 * - CARTS-compatible: Changed from VLA pointer casts to explicit
 *   array-of-arrays allocation pattern
 */

 #include <stdlib.h>

 /*
  * Simplified sweep function using direct parallel for loops
  * instead of parallel region with nested for loops.
  * Now uses explicit 2D arrays instead of VLA casts.
  */
 static void sweep(int nx, int ny, double dx, double dy, double **f, int itold,
                   int itnew, double **u, double **unew, int block_size) {
   int i, j, it;

   for (it = itold + 1; it <= itnew; it++) {
     /* Save the current estimate */
#pragma omp parallel for private(j)
     for (i = 0; i < nx; i++) {
       for (j = 0; j < ny; j++) {
         u[i][j] = unew[i][j];
       }
     }

     /* Compute a new estimate */
#pragma omp parallel for private(j)
     for (i = 0; i < nx; i++) {
       for (j = 0; j < ny; j++) {
         if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
           unew[i][j] = f[i][j];
         } else {
           unew[i][j] =
               0.25 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] +
                       u[i + 1][j] + f[i][j] * dx * dy);
         }
       }
     }
   }
 }

int main(void) {
#ifdef SIZE
  int nx = SIZE, ny = SIZE;
#else
  int nx = 100, ny = 100; // default for testing
#endif
  int itold = 0, itnew = 10;
  int block_size = 10;
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  /* Allocate 2D arrays using array-of-arrays pattern (CARTS-compatible) */
  double **f = (double **)malloc(nx * sizeof(double *));
  double **u = (double **)malloc(nx * sizeof(double *));
  double **unew = (double **)malloc(nx * sizeof(double *));

  for (int i = 0; i < nx; i++) {
    f[i] = (double *)malloc(ny * sizeof(double));
    u[i] = (double *)malloc(ny * sizeof(double));
    unew[i] = (double *)malloc(ny * sizeof(double));
  }

  /* Initialize arrays */
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      f[i][j] = 0.0;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
    }
  }

  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);

  /* Free 2D arrays */
  for (int i = 0; i < nx; i++) {
    free(f[i]);
    free(u[i]);
    free(unew[i]);
  }
  free(f);
  free(u);
  free(unew);

  return 0;
}
