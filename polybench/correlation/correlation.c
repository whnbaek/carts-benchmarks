#include <math.h>
#include <stdio.h>

#include "correlation.h"

static void print_array(int m, int n,
                        DATA_TYPE POLYBENCH_2D(corr, M, M, m, n),
                        const char *name) {
  (void)name;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, corr[i][j]);
      if ((i * n + j) % 20 == 0) {
        fprintf(stderr, "\n");
      }
    }
  }
  fprintf(stderr, "\n");
}

static void init_array(int m, int n,
                       DATA_TYPE POLYBENCH_2D(data, M, N, m, n)) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      data[i][j] = (DATA_TYPE)(i * j) / (DATA_TYPE)M + (DATA_TYPE)i / (j + 1);
    }
  }
}

static void kernel_correlation(
    int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
    DATA_TYPE POLYBENCH_2D(corr, M, M, m, m),
    DATA_TYPE POLYBENCH_1D(mean, M, m),
    DATA_TYPE POLYBENCH_1D(stddev, M, m)) {
  /* Step 1: compute mean of each row. */
  for (int i = 0; i < m; i++) {
    mean[i] = 0.0;
    for (int j = 0; j < n; j++) {
      mean[i] += data[i][j];
    }
    mean[i] /= FLOAT_N;
  }

  /* Step 2: compute std deviation of each row. */
  for (int i = 0; i < m; i++) {
    stddev[i] = 0.0;
    for (int j = 0; j < n; j++) {
      DATA_TYPE diff = data[i][j] - mean[i];
      stddev[i] += diff * diff;
    }
    stddev[i] = sqrt(stddev[i] / FLOAT_N);
    if (stddev[i] <= EPS) {
      stddev[i] = 1.0;
    }
  }

  /* Step 3: center and scale the data. */
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      data[i][j] =
          (data[i][j] - mean[i]) / (sqrt(FLOAT_N) * stddev[i]);
    }
  }

  /* Step 4: compute the correlation matrix. */
  for (int i = 0; i < m; i++) {
    corr[i][i] = 1.0;
    for (int j = i + 1; j < m; j++) {
      DATA_TYPE sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += data[i][k] * data[j][k];
      }
      corr[i][j] = sum;
      corr[j][i] = sum;
    }
  }
}

int main(int argc, char **argv) {
  int m = M;
  int n = N;

  POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(corr, DATA_TYPE, M, M, m, m);
  POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(stddev, DATA_TYPE, M, m);

  init_array(m, n, POLYBENCH_ARRAY(data));

  kernel_correlation(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(corr),
                     POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev));

  polybench_prevent_dce(print_array(m, m, POLYBENCH_ARRAY(corr), "corr"));

  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
