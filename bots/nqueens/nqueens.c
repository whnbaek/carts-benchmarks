/* BOTS-like N-Queens using OpenMP tasks (single-file) */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 14
#endif

static int is_safe(int *board, int row, int col) {
  for (int i = 0; i < row; i++) {
    int c = board[i];
    if (c == col || abs(c - col) == (row - i)) return 0;
  }
  return 1;
}

static void solve_task(int *board, int row, long *solutions) {
  if (row == N) {
    #pragma omp atomic
    (*solutions)++;
    return;
  }
  for (int col = 0; col < N; col++) {
    if (is_safe(board, row, col)) {
      board[row] = col;
      if (row < 6) {
        #pragma omp task firstprivate(row) depend(inout: solutions[0])
        {
          int local[N];
          for (int i = 0; i <= row; i++) local[i] = board[i];
          solve_task(local, row + 1, solutions);
        }
      } else {
        solve_task(board, row + 1, solutions);
      }
    }
  }
}

int main(void) {
  long solutions = 0;
  int board[N];
  for (int i = 0; i < N; i++) board[i] = -1;

  #pragma omp parallel
  {
    #pragma omp single nowait
    solve_task(board, 0, &solutions);
  }

  printf("solutions=%ld\n", solutions);
  return 0;
}


