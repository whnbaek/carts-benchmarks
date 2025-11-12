/**********************************************************************************************/
/*  Standalone CARTS test version of SparseLU with tasks
 *  Self-contained single-file version for CARTS compiler testing
 *
 *  Based on Barcelona OpenMP Tasks Suite
 *  Copyright (C) 2009 Barcelona Supercomputing Center
 *  License: GNU GPL
 */
/**********************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 1.0E-6

//===----------------------------------------------------------------------===//
// Utility Functions (Static)
//===----------------------------------------------------------------------===//

static float *allocate_clean_block(int submatrix_size) {
  int i, j;
  float *p, *q;

  p = (float *)malloc(submatrix_size * submatrix_size * sizeof(float));
  q = p;
  if (p != NULL) {
    for (i = 0; i < submatrix_size; i++)
      for (j = 0; j < submatrix_size; j++) {
        (*p) = 0.0;
        p++;
      }
  } else {
    fprintf(stderr, "Error: malloc failed in allocate_clean_block\n");
    exit(101);
  }
  return (q);
}

static void lu0(float *diag, int submatrix_size) {
  int i, j, k;

  for (k = 0; k < submatrix_size; k++)
    for (i = k + 1; i < submatrix_size; i++) {
      diag[i * submatrix_size + k] =
          diag[i * submatrix_size + k] / diag[k * submatrix_size + k];
      for (j = k + 1; j < submatrix_size; j++)
        diag[i * submatrix_size + j] =
            diag[i * submatrix_size + j] -
            diag[i * submatrix_size + k] * diag[k * submatrix_size + j];
    }
}

static void bdiv(float *diag, float *row, int submatrix_size) {
  int i, j, k;
  for (i = 0; i < submatrix_size; i++)
    for (k = 0; k < submatrix_size; k++) {
      row[i * submatrix_size + k] =
          row[i * submatrix_size + k] / diag[k * submatrix_size + k];
      for (j = k + 1; j < submatrix_size; j++)
        row[i * submatrix_size + j] =
            row[i * submatrix_size + j] -
            row[i * submatrix_size + k] * diag[k * submatrix_size + j];
    }
}

static void bmod(float *row, float *col, float *inner, int submatrix_size) {
  int i, j, k;
  for (i = 0; i < submatrix_size; i++)
    for (j = 0; j < submatrix_size; j++)
      for (k = 0; k < submatrix_size; k++)
        inner[i * submatrix_size + j] =
            inner[i * submatrix_size + j] -
            row[i * submatrix_size + k] * col[k * submatrix_size + j];
}

static void fwd(float *diag, float *col, int submatrix_size) {
  int i, j, k;
  for (j = 0; j < submatrix_size; j++)
    for (k = 0; k < submatrix_size; k++)
      for (i = k + 1; i < submatrix_size; i++)
        col[i * submatrix_size + j] =
            col[i * submatrix_size + j] -
            diag[i * submatrix_size + k] * col[k * submatrix_size + j];
}

//===----------------------------------------------------------------------===//
// Matrix Generation
//===----------------------------------------------------------------------===//

static void genmat(float *M[], int matrix_size, int submatrix_size) {
  int null_entry, init_val, i, j, ii, jj;

  init_val = 1325;

  /* generating the structure */
  for (ii = 0; ii < matrix_size; ii++) {
    for (jj = 0; jj < matrix_size; jj++) {
#pragma omp task shared(M)
      {
        float *p;
        /* computing null entries */
        null_entry = 0;
        if ((ii < jj) && (ii % 3 != 0))
          null_entry = 1;
        if ((ii > jj) && (jj % 3 != 0))
          null_entry = 1;
        if (ii % 2 == 1)
          null_entry = 1;
        if (jj % 2 == 1)
          null_entry = 1;
        if (ii == jj)
          null_entry = 0;
        if (ii == jj - 1)
          null_entry = 0;
        if (ii - 1 == jj)
          null_entry = 0;
        /* allocating matrix */
        if (null_entry == 0) {
          M[ii * matrix_size + jj] =
              (float *)malloc(submatrix_size * submatrix_size * sizeof(float));
          if (M[ii * matrix_size + jj] == NULL) {
            fprintf(stderr, "Error: malloc failed in genmat\n");
            exit(101);
          }
          /* initializing matrix */
          p = M[ii * matrix_size + jj];
          for (i = 0; i < submatrix_size; i++) {
            for (j = 0; j < submatrix_size; j++) {
              init_val = (3125 * init_val) % 65536;
              (*p) = (float)((init_val - 32768.0) / 16384.0);
              p++;
            }
          }
        } else {
          M[ii * matrix_size + jj] = NULL;
        }
      }
    }
  }
#pragma omp taskwait
}

static void sparselu_init(float ***pBENCH, int matrix_size, int submatrix_size) {
  *pBENCH = (float **)malloc(matrix_size * matrix_size * sizeof(float *));
  if (*pBENCH == NULL) {
    fprintf(stderr, "Error: malloc failed for benchmark matrix\n");
    exit(101);
  }
  genmat(*pBENCH, matrix_size, submatrix_size);
}

//===----------------------------------------------------------------------===//
// Parallel SparseLU with Tasks
//===----------------------------------------------------------------------===//

static void sparselu_par_call(float **BENCH, int matrix_size, int submatrix_size) {
  int ii, jj, kk;

#pragma omp parallel
#pragma omp single nowait
  for (kk = 0; kk < matrix_size; kk++) {
    lu0(BENCH[kk * matrix_size + kk], submatrix_size);
    for (jj = kk + 1; jj < matrix_size; jj++)
      if (BENCH[kk * matrix_size + jj] != NULL)
#pragma omp task untied firstprivate(kk, jj) shared(BENCH)
      {
        fwd(BENCH[kk * matrix_size + kk], BENCH[kk * matrix_size + jj],
            submatrix_size);
      }
    for (ii = kk + 1; ii < matrix_size; ii++)
      if (BENCH[ii * matrix_size + kk] != NULL)
#pragma omp task untied firstprivate(kk, ii) shared(BENCH)
      {
        bdiv(BENCH[kk * matrix_size + kk], BENCH[ii * matrix_size + kk],
             submatrix_size);
      }

#pragma omp taskwait

    for (ii = kk + 1; ii < matrix_size; ii++)
      if (BENCH[ii * matrix_size + kk] != NULL)
        for (jj = kk + 1; jj < matrix_size; jj++)
          if (BENCH[kk * matrix_size + jj] != NULL)
#pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
          {
            if (BENCH[ii * matrix_size + jj] == NULL)
              BENCH[ii * matrix_size + jj] =
                  allocate_clean_block(submatrix_size);
            bmod(BENCH[ii * matrix_size + kk], BENCH[kk * matrix_size + jj],
                 BENCH[ii * matrix_size + jj], submatrix_size);
          }

#pragma omp taskwait
  }
}

//===----------------------------------------------------------------------===//
// Main Test Function
//===----------------------------------------------------------------------===//

int main(void) {
  float **BENCH;
  int matrix_size = 16;      // Small size for testing
  int submatrix_size = 8;    // Subblock size

  printf("SparseLU Task Test (CARTS)\n");
  printf("Matrix size: %d x %d blocks\n", matrix_size, matrix_size);
  printf("Submatrix size: %d x %d\n", submatrix_size, submatrix_size);

  // Initialize matrix
#pragma omp parallel
#pragma omp master
  sparselu_init(&BENCH, matrix_size, submatrix_size);

  printf("Running parallel SparseLU with tasks...\n");

  // Run parallel SparseLU
  sparselu_par_call(BENCH, matrix_size, submatrix_size);

  printf("SparseLU completed successfully!\n");

  // Cleanup
  for (int i = 0; i < matrix_size * matrix_size; i++) {
    if (BENCH[i] != NULL)
      free(BENCH[i]);
  }
  free(BENCH);

  return 0;
}
