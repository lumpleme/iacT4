#include "matrix_lib.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

unsigned long int threads_per_block = 256;
unsigned long int blocks_per_grid = 1024;

__host__
int cuda_check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

__host__
int matrix_to_host(struct matrix *matrix) {
    size_t bytes = (size_t)matrix->height * (size_t)matrix->width * sizeof(float);
    cudaError_t err = cudaMemcpy(matrix->h_rows, matrix->d_rows, bytes, cudaMemcpyDeviceToHost);
    if (!cuda_check(err, "cudaMemcpy D->H")) return 0;
    return 1;
}

//-------------------- Kernels ------------------------

__host__
void set_vars(unsigned long int tpb, unsigned long int bpg) {
    if (tpb <= 1024){
    threads_per_block = tpb;
  }
  if (bpg <= 2147483647){
    blocks_per_grid = bpg;
  }
}

__global__ 
void scalar_kernel(float scalar, float *d_rows, unsigned long n) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long stride = gridDim.x * blockDim.x;
  for (unsigned long i = idx; i < n; i += stride) {
    d_rows[i] = d_rows[i] * scalar;
  }
}


__global__ 
void matmul_kernel(const float *A, const float *B, float *C,
                             int A_height, int A_width, int B_width) {
  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long total = (unsigned long)A_height * (unsigned long)B_width;
  if (idx >= total) return;
  int row = idx / B_width;
  int col = idx % B_width;
  float sum = 0.0f;
  for (int k = 0; k < A_width; ++k) {
    sum += A[(size_t)row * A_width + k] * B[(size_t)k * B_width + col];
  }
  C[(size_t)row * B_width + col] = sum;
}

__global__
void matmul_row_kernel(const float *A_row, const float *B, float *C_row,
                       int A_width, int B_width) {
  unsigned long col = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= (unsigned long)B_width) return;
  float sum = 0.0f;
  for (int k = 0; k < A_width; ++k) {
    sum += A_row[k] * B[(size_t)k * B_width + col];
  }
  C_row[col] = sum;
}


//-------------------- Funções ------------------------
__host__
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
  if (matrix == NULL || matrix->h_rows == NULL || matrix->d_rows == NULL) return 0;
  
  unsigned long n = matrix->height * matrix->width;
  if (n == 0) return 0;

  unsigned long int threads = threads_per_block;

  cudaError_t err;
  // Full allocation on device: operate on full buffer
  if (matrix->alloc_mode == 1) {
    if (matrix->d_rows == NULL) return 0;
    unsigned long blocks = (n + threads - 1) / threads;
    if (blocks > blocks_per_grid) blocks = blocks_per_grid;

    scalar_kernel<<<(int)blocks, threads>>>(scalar_value, matrix->d_rows, n);
    err = cudaDeviceSynchronize();
    if (!cuda_check(err, "cudaDeviceSynchronize (scalar_matrix_mult)")) return 0;
    if (!matrix_to_host(matrix)) return 0;
    return 1;
  }
  else{
    if (matrix->d_rows == NULL) return 0;
    unsigned long row_n = matrix->width;
    unsigned long blocks = (row_n + threads - 1) / threads;
    size_t offset = 0;
    size_t bytes = (size_t)matrix->width * sizeof(float);
    if (blocks > blocks_per_grid) blocks = blocks_per_grid;
    

    // process row by row
    for (unsigned long r = 0; r < matrix->height; ++r) {
      scalar_kernel<<<(int)blocks, threads>>>(scalar_value, matrix->d_rows, row_n);
      err = cudaDeviceSynchronize();
      if (!cuda_check(err, "cudaDeviceSynchronize (scalar row)")) return 0;

      err = cudaMemcpy(matrix->h_rows + offset, matrix->d_rows, bytes, cudaMemcpyDeviceToHost);
      if (!cuda_check(err, "cudaMemcpy D->H (scalar row)")) return 0;

      offset = (size_t)r * matrix->width;
      err = cudaMemcpy(matrix->d_rows, matrix->h_rows + offset, bytes, cudaMemcpyHostToDevice);
      if (!cuda_check(err, "cudaMemcpy H->D (scalar row)")) return 0;
    }
    return 1;
  }

  return 0;
}

__host__
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
  if (matrixA == NULL || matrixB == NULL || matrixC == NULL) return 0;
  if (matrixA->width != matrixB->height) return 0;
  if (matrixC->height != matrixA->height || matrixC->width != matrixB->width) return 0;
  if (matrixA->h_rows == NULL || matrixB->h_rows == NULL || matrixC->h_rows == NULL) return 0;
  if (matrixA->d_rows == NULL || matrixB->d_rows == NULL || matrixC->d_rows == NULL) return 0;


  int A_h = (int)matrixA->height;
  int A_w = (int)matrixA->width;
  int B_w = (int)matrixB->width;

  cudaError_t err;

  unsigned long total = (unsigned long)matrixC->height * (unsigned long)matrixC->width;
  if (total == 0) return 0;

  unsigned long int threads = threads_per_block;

  // If A and C are fully allocated on device, we can run the full kernel
  if (matrixA->alloc_mode == 1 && matrixC->alloc_mode == 1) {
    if (matrixA->d_rows == NULL || matrixB->d_rows == NULL) return 0;

    // Allocate device memory for C if needed
    size_t bytesC = (size_t)matrixC->height * (size_t)matrixC->width * sizeof(float);
    if (matrixC->d_rows == NULL) {
      err = cudaMalloc((void**)&matrixC->d_rows, bytesC);
      if (!cuda_check(err, "cudaMalloc (matrixC)")) return 0;
    }

    unsigned long blocks = (total + threads - 1) / threads;
    if (blocks > blocks_per_grid) blocks = blocks_per_grid;

    matmul_kernel<<<(int)blocks, threads>>>(matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, A_h, A_w, B_w);
    err = cudaDeviceSynchronize();
    if (!cuda_check(err, "cudaDeviceSynchronize (matrix_matrix_mult)")) return 0;

    if (!matrix_to_host(matrixC)) return 0;
    return 1;
  }

  // Partial mode: process row-by-row. Expect B to be fully on device.
  if (matrixB->d_rows == NULL) return 0;

  // Ensure matrixA.d_rows and matrixC.d_rows are allocated as per-row buffers
  if (matrixA->alloc_mode == 0 && matrixA->d_rows == NULL) return 0;
  if (matrixC->alloc_mode == 0 && matrixC->d_rows == NULL) return 0;

  unsigned long blocks_row = ((unsigned long)B_w + threads - 1) / threads;
  if (blocks_row > blocks_per_grid) blocks_row = blocks_per_grid;

  for (unsigned long r = 0; r < (unsigned long)matrixA->height; ++r) {
    // copy A row r to device buffer (or get pointer if A is full on device)
    const float *A_row_device_ptr = NULL;
    if (matrixA->alloc_mode == 0) {
      size_t bytes_row = (size_t)A_w * sizeof(float);
      err = cudaMemcpy(matrixA->d_rows, matrixA->h_rows + (size_t)r * A_w, bytes_row, cudaMemcpyHostToDevice);
      if (!cuda_check(err, "cudaMemcpy H->D A row")) return 0;
      A_row_device_ptr = matrixA->d_rows;
    } else {
      A_row_device_ptr = matrixA->d_rows + (size_t)r * A_w;
    }

    // compute C row into C.d_rows (device buffer) or into full C.d_rows offset
    float *C_row_device_ptr = NULL;
    if (matrixC->alloc_mode == 0) {
      C_row_device_ptr = matrixC->d_rows; // per-row buffer
    } else {
      C_row_device_ptr = matrixC->d_rows + (size_t)r * B_w;
    }

    matmul_row_kernel<<<(int)blocks_row, threads>>>(A_row_device_ptr, matrixB->d_rows, C_row_device_ptr, A_w, B_w);
    err = cudaDeviceSynchronize();
    if (!cuda_check(err, "cudaDeviceSynchronize (matmul row)")) return 0;

    // copy C row back to host if C is partial; if C is full, we'll copy full later
    if (matrixC->alloc_mode == 0) {
      size_t bytes_row = (size_t)B_w * sizeof(float);
      err = cudaMemcpy(matrixC->h_rows + (size_t)r * B_w, C_row_device_ptr, bytes_row, cudaMemcpyDeviceToHost);
      if (!cuda_check(err, "cudaMemcpy D->H C row")) return 0;
    }
  }

  // If C was fully allocated on device, copy full matrix back now
  if (matrixC->alloc_mode == 1) {
    if (!matrix_to_host(matrixC)) return 0;
  }

  return 1;
}
