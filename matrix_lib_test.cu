#include "matrix_lib.h"
extern "C" {
#include "timer.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static inline int cuda_check_err(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    return 0;
  }
  return 1;
}

void print_matrix(struct matrix M) {
  for (int i = 0; i < 16 && i < M.height; i++){
    for (int j = 0; j < 16 && j < M.width; j++){
      printf("%.1f ", M.h_rows[i * M.width + j]);
    }
    printf("\n");
  }
  if (M.height > 15 && M.width > 15) {
    printf("Limite de 256 elementos impressos.\n");
  }
}

int main(int argc, char* argv[]) {
  struct matrix A, B, C;
  struct timeval start, stop, overall_t1, overall_t2;

  // Overall timer start
  gettimeofday(&overall_t1, NULL);

  if (argc != 13) {
    printf("Usage: %s <scalar_value> <matrix1_height> <matrix1_width> <matrix2_height> <matrix2_width> <threads_block> <blocks_grid> <max_MiB_gpu> <matrix1_binfile> <matrix2_binfile> <result1_binfile> <result2_binfile>\n", argv[0]);
    return 0;
  }

  A.height = strtoul(argv[2], NULL, 10);
  A.width = strtoul(argv[3], NULL, 10);
  A.h_rows = (float*) aligned_alloc(32, sizeof(float) * A.height * A.width);
  if (A.h_rows == NULL) {
    printf("%s: vector allocation problem.\n", argv[0]);
    return 1;
  }
  A.d_rows = NULL;

  B.height = strtoul(argv[4], NULL, 10);
  B.width = strtoul(argv[5], NULL, 10);
  B.h_rows = (float*) aligned_alloc(32, sizeof(float) * B.height * B.width);
  if (B.h_rows == NULL) {
    printf("%s: vector allocation problem.\n", argv[0]);
    return 1;
  }
  B.d_rows = NULL;

  int threads_per_block = strtoul(argv[6], NULL, 10);
  int blocks_per_grid = strtoul(argv[7], NULL, 10);
  int max_MiB_gpu = strtoul(argv[8], NULL, 10);

  set_vars(threads_per_block, blocks_per_grid);

  FILE *file = fopen(argv[9], "rb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }

  for (unsigned long int i = 0; i < A.height; i++) {
    for (unsigned long int j = 0; j < A.width; j++) {
      fread(&A.h_rows[i * A.width + j], sizeof(float), 1, file);
    }
  }
  fclose(file);

  file = fopen(argv[10], "rb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }
  for (unsigned long int i = 0; i < B.height; i++) {
    for (unsigned long int j = 0; j < B.width; j++) {
      fread(&B.h_rows[i * B.width + j], sizeof(float), 1, file);
    }
  }
  fclose(file);

  C.height = A.height;
  C.width = B.width;
  C.h_rows = (float*) aligned_alloc(32, sizeof(float) * C.height * C.width);
  if (C.h_rows == NULL) {
    printf("%s: vector allocation problem.\n", argv[0]);
    return 1;
  }
  C.d_rows = NULL;

  //calculo
  
  unsigned long int n_bytes = (size_t)max_MiB_gpu * 1024 * 1024;

  size_t bytesA = (size_t)A.height * (size_t)A.width * sizeof(float);
  size_t bytesB = (size_t)B.height * (size_t)B.width * sizeof(float);
  size_t bytesC = (size_t)C.height * (size_t)C.width * sizeof(float);

  cudaError_t err;

  if (bytesA + bytesB + bytesC > n_bytes) {
    if (bytesB + ((size_t)A.width + (size_t)C.width) * sizeof(float) > n_bytes) {
        printf("Matrix B is too large for the GPU memory limit.\n");
        return 1;
    }

    err = cudaMalloc((void**)&A.d_rows, (size_t)A.width * sizeof(float));
    if (!cuda_check_err(err, "cudaMalloc A.d_rows")) return 1;
    err = cudaMemcpy(A.d_rows, A.h_rows, bytesA, cudaMemcpyHostToDevice);
    if (!cuda_check_err(err, "cudaMemcpy H->D A")) return 1; 
    A.alloc_mode = 0; //partial

    err = cudaMalloc((void**)&C.d_rows, (size_t)C.width * sizeof(float));
    if (!cuda_check_err(err, "cudaMalloc C.d_rows")) return 1;
    err = cudaMemcpy(C.d_rows, C.h_rows, bytesC, cudaMemcpyHostToDevice);
    if (!cuda_check_err(err, "cudaMemcpy H->D C")) return 1;
    C.alloc_mode = 0; //partial
    
  }
  else {
    err = cudaMalloc((void**)&A.d_rows, bytesA);
    if (!cuda_check_err(err, "cudaMalloc A.d_rows")) return 1;
    err = cudaMemcpy(A.d_rows, A.h_rows, bytesA, cudaMemcpyHostToDevice);
    if (!cuda_check_err(err, "cudaMemcpy H->D A")) return 1; 
    A.alloc_mode = 1; //full

    err = cudaMalloc((void**)&C.d_rows, bytesC);
    if (!cuda_check_err(err, "cudaMalloc C.d_rows")) return 1;
    err = cudaMemcpy(C.d_rows, C.h_rows, bytesC, cudaMemcpyHostToDevice);
    if (!cuda_check_err(err, "cudaMemcpy H->D C")) return 1;
    C.alloc_mode = 1; //full
  }
    // Always fully alocated 
    err = cudaMalloc((void**)&B.d_rows, bytesB);
    if (!cuda_check_err(err, "cudaMalloc B.d_rows")) return 1;
    err = cudaMemcpy(B.d_rows, B.h_rows, bytesB, cudaMemcpyHostToDevice);
    if (!cuda_check_err(err, "cudaMemcpy H->D B")) return 1;
    B.alloc_mode = 1; //full
  

  printf("---------- Matrix A ----------\n");
  print_matrix(A);

  printf("---------- Matrix B ----------\n");
  print_matrix(B);

  // Scalar multiply on GPU
  gettimeofday(&start, NULL);

  if (!scalar_matrix_mult(strtof(argv[1], NULL), &A)) {
    fprintf(stderr, "scalar_matrix_mult failed\n");
    return 1;
  }
  gettimeofday(&stop, NULL);
  printf("Scalar_matrix_mult (GPU) time: %f ms\n", timedifference_msec(start, stop));

  printf("---------- Scalar x Matrix A (GPU result) ----------\n");
  print_matrix(A);

  // Save scalar result to file
  file = fopen(argv[11], "wb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }
  for (unsigned long int i = 0; i < A.height; i++) {
    for (unsigned long int j = 0; j < A.width; j++) {
      fwrite(&A.h_rows[i * A.width + j], sizeof(float), 1, file);
    }
  }
  fclose(file);

  // Matrix multiplication on GPU
  gettimeofday(&start, NULL);

  if (!matrix_matrix_mult(&A, &B, &C)) {
    fprintf(stderr, "matrix_matrix_mult failed\n");
    return 1;
  }
  gettimeofday(&stop, NULL);


  printf("Matrix_matrix_mult (GPU) time: %f ms\n", timedifference_msec(start, stop));

  printf("---------- Matrix C (GPU result) ----------\n");
  print_matrix(C);

  // Save C to file
  file = fopen(argv[12], "wb");
  if (file == NULL) {
    printf("%s: file open error.\n", argv[0]);
    return 1;
  }
  for (unsigned long int i = 0; i < C.height; i++) {
    for (unsigned long int j = 0; j < C.width; j++) {
      fwrite(&C.h_rows[i * C.width + j], sizeof(float), 1, file);
    }
  }
  fclose(file);

  // Free device memory and host buffers (we allocated device memory in this test)
  if (A.d_rows){
    cudaFree(A.d_rows);
  }
  if (B.d_rows){
    cudaFree(B.d_rows);
  }
  if (C.d_rows){
    cudaFree(C.d_rows);
  }

  free(A.h_rows);
  free(B.h_rows);
  free(C.h_rows);

  gettimeofday(&overall_t2, NULL);
  printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

  system("lscpu | grep 'Model name'");

  return 0;
}
