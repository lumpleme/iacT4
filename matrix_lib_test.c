// Pedro Goncalves Mannarino - 2210617
// Luiza Marcondes Paes Leme - 2210275

#include "matrix_lib.h"
#include "timer.h"

void print_matrix(struct matrix M) {

  for (int i = 0; i < 16 && i < M.height; i++){
    for (int j = 0; j < 16 && j < M.width; j++){
      printf("%.1f ", M.rows[i * M.width + j]);
    }
      printf("\n");
    }
    if (M.height > 15 && M.width > 15) {
      printf("Limite de 256 elementos impressos.\n");
  }

  return;
}

int main(int argc, char* argv[]) {
  struct matrix A, B, C;
  struct timeval start, stop, overall_t1, overall_t2;

  // Overall timer início
  gettimeofday(&overall_t1, NULL);

  if (argc != 11) {
    printf("Usage: %s <scalar_value> <matrix1_height> <matrix1_width> <matrix2_height> <matrix2_width> <n_threads> <matrix1_binfile> <matrix2_binfile> <result1_binfile> <result2_binfile>\n", argv[0]);
    return 0;
  }

  A.height = strtoul(argv[2], NULL, 10);
  A.width = strtoul(argv[3], NULL, 10);
  A.rows = (float*) aligned_alloc(32, sizeof(float) * A.height * A.width);
  if (A.rows == NULL) {
    printf("%s: vector allocation problem.\n", argv[0]);
    return 1;
  }

  B.height = strtoul(argv[4], NULL, 10);
  B.width = strtoul(argv[5], NULL, 10);
  B.rows = (float*) aligned_alloc(32, sizeof(float) * B.height * B.width);
  if (B.rows == NULL) {
    printf("%s: vector allocation problem.\n", argv[0]);
    return 1;
  }

  int n_threads = atoi(argv[6]);

  FILE *file = fopen(argv[7], "rb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }

  for (unsigned long int i = 0; i < A.height; i++) {
    for (unsigned long int j = 0; j < A.width; j++) {
      fread(&A.rows[i * A.width + j], sizeof(float), 1, file);
    }
  }

  fclose(file);

  file = fopen(argv[8], "rb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }

  for (unsigned long int i = 0; i < B.height; i++) {
    for (unsigned long int j = 0; j < B.width; j++) {
      fread(&B.rows[i * B.width + j], sizeof(float), 1, file);
    }
  }

  fclose(file);

  C.height = A.height;
  C.width = B.width;
  C.rows = (float*) aligned_alloc(32, sizeof(float) * C.height * C.width);

  if (C.rows == NULL) {
    printf("%s: vector allocation problem.\n", argv[0]);
    return 1;
  }

  printf("---------- Matrix A ----------\n");
  print_matrix(A);

  printf("---------- Matrix B ----------\n");
  print_matrix(B);

  gettimeofday(&start, NULL);

  scalar_matrix_mult(strtof(argv[1], NULL), &A, n_threads);

  gettimeofday(&stop, NULL);

  printf("Scalar_matrix_mult time: %f ms\n", timedifference_msec(start, stop));

  printf("---------- Scalar x Matrix A ----------\n");
  print_matrix(A);

  file = fopen(argv[9], "wb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }

  for (unsigned long int i = 0; i < A.height; i++) {
    for (unsigned long int j = 0; j < A.width; j++) {
      fwrite(&A.rows[i * A.width + j], sizeof(float), 1, file);
    }
  }

  fclose(file);


  gettimeofday(&start, NULL);

  matrix_matrix_mult(&A, &B, &C, n_threads);

  gettimeofday(&stop, NULL);

  printf("Matrix_matrix_mult time: %f ms\n", timedifference_msec(start, stop));

  printf("---------- Matrix C ----------\n");
  print_matrix(C);

  file = fopen(argv[10], "wb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }

  for (unsigned long int i = 0; i < C.height; i++) {
    for (unsigned long int j = 0; j < C.width; j++) {
      fwrite(&C.rows[i * C.width + j], sizeof(float), 1, file);
    }
  }

  fclose(file);

  gettimeofday(&overall_t2, NULL);

  printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));
  
  system("lscpu | grep 'Model name'");

  return 0;
}
