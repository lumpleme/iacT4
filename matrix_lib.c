#include "matrix_lib.h"
#include <immintrin.h>
#include <cpuid.h>
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>

struct arguments{
  struct matrix *matrixA;
  struct matrix *matrixB;
  struct matrix *matrixC;
  unsigned long int i;
  bool *flag;
  float scalar;
};

// ---------------------------------- THREADS ---------------------------------- //

void *InitLine(void *argss){
  struct arguments *args = (struct arguments *) argss;

  unsigned long int line = args->i * args->matrixC->width;
  
  for (unsigned long int j = 0; j < args->matrixC->width; j++){
    args->matrixC->rows[line + j] = 0.0;
  }
  
  *(args->flag) = false;
  free(args);
  pthread_exit(NULL);
}

void *ScalarMultLine(void *argss) {
  struct arguments *args = (struct arguments *) argss;
  
  __m256 scalar_256 = _mm256_set1_ps(args->scalar);
  __m256 array_256, result_256;
  float *address;
  
  unsigned long int line = args->i * args->matrixA->width;
  
  for (unsigned long int j = 0; j < args->matrixA->width / 8; j++){
    address = args->matrixA->rows + line + j * 8;
    
    array_256 = _mm256_load_ps(address);
    
    result_256 = _mm256_mul_ps(array_256, scalar_256);
    
    _mm256_store_ps(address, result_256);
  }
  
  *(args->flag) = false;
  free(args);
  pthread_exit(NULL);
}

void *MatrixMultLine(void *argss){
  struct arguments *args = (struct arguments *) argss;

  __m256 result_256, arrayA_256, arrayB_256;
  float *addressA, *addressB, *addressC;
  unsigned long int i = args->i;
  
  for (unsigned long int j = 0; j < args->matrixB->height; j++){    
    addressA = args->matrixA->rows + i * args->matrixA->width + j;
        
    arrayA_256 = _mm256_set1_ps(*addressA);
  
    for (unsigned long int k = 0; k < args->matrixB->width / 8; k++){      
      addressC = args->matrixC->rows + i * args->matrixC->width + k * 8;
      
      result_256 = _mm256_load_ps(addressC);
      
      addressB = args->matrixB->rows + j * args->matrixB->width + k * 8;
      
      arrayB_256 = _mm256_load_ps(addressB);
      
      result_256 = _mm256_fmadd_ps(arrayA_256, arrayB_256, result_256);
          
      _mm256_store_ps(addressC, result_256);
    }
  }
    
  *(args->flag) = false;
  free(args);
  pthread_exit(NULL);
}

// ---------------------------------- FUNÇÕES ---------------------------------- //

int scalar_matrix_mult(float scalar_value, struct matrix *matrix, int n_threads){
  if (matrix == NULL){
    return 0;
  }
  
  pthread_t threads[n_threads];
  bool thread_flags[n_threads];
  
  for (int i = 0; i < n_threads; i++){
    thread_flags[i] = false;
  }

  int rc;
  long t = 0;
  
  for (unsigned long int i = 0; i < matrix->height; i++) {
    // Allocate new arguments for each thread
    struct arguments *args = malloc(sizeof(struct arguments));
    if (args == NULL) {
      printf("Error: unable to allocate memory\n");
      return 0;
    }
    
    args->matrixA = matrix;
    args->i = i;
    args->scalar = scalar_value;
    
    while (true) {
      if (!thread_flags[t]) {
        thread_flags[t] = true;
        args->flag = &thread_flags[t];
        
        rc = pthread_create(&threads[t], NULL, ScalarMultLine, (void*)args);
        if (rc){
          printf("Error: unable to create thread, %d\n", rc);
          free(args);
          exit(0);
        }
        
        t = 0;
        break;
      }
      
      t++;
      if (t >= n_threads) {
        t = 0;
      }
    }
  }

  for (int i = 0; i < n_threads; i++){
    if (thread_flags[i]) {
      rc = pthread_join(threads[i], NULL);
      if (rc){
        printf("Error: unable to join, %d\n", rc);
        exit(0);
      }
    }
  }
  
  return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC, int n_threads){
  if (matrixA == NULL || matrixB == NULL || matrixC == NULL){
    return 0;
  }
  if (matrixA->width != matrixB->height){
    return 0;
  }
  if (matrixC->height != matrixA->height || matrixC->width != matrixB->width){
    return 0;
  }

  pthread_t threads[n_threads];
  bool thread_flags[n_threads];
  
  for (int i = 0; i < n_threads; i++){
    thread_flags[i] = false;
  }

  int rc;
  long t = 0;
  
  // Initialize matrix C with zeros
  for (unsigned long int i = 0; i < matrixC->height; i++) {
    struct arguments *args = malloc(sizeof(struct arguments));
    if (args == NULL) {
      printf("Error: unable to allocate memory\n");
      return 0;
    }
    
    args->matrixC = matrixC;
    args->i = i;
    
    while (true) {
      if (!thread_flags[t]) {
        thread_flags[t] = true;
        args->flag = &thread_flags[t];
        
        rc = pthread_create(&threads[t], NULL, InitLine, (void*)args);
        if (rc){
          printf("Error: unable to create thread, %d\n", rc);
          free(args);
          exit(0);
        }
        
        t = 0;
        break;
      }
      
      t++;
      if (t >= n_threads) {
        t = 0;
      }
    }
  }

  // Wait for all initialization threads to complete
  for (int i = 0; i < n_threads; i++){
    if (thread_flags[i]) {
      rc = pthread_join(threads[i], NULL);
      if (rc){
        printf("Error: unable to join, %d\n", rc);
        exit(0);
      }
    }
  }
  
  // Reset flags for the multiplication phase
  for (int i = 0; i < n_threads; i++){
    thread_flags[i] = false;
  }
  
  // Matrix multiplication
  for (unsigned long int i = 0; i < matrixA->height; i++){
    struct arguments *args = malloc(sizeof(struct arguments));
    if (args == NULL) {
      printf("Error: unable to allocate memory\n");
      return 0;
    }
    
    args->matrixA = matrixA;
    args->matrixB = matrixB;
    args->matrixC = matrixC;
    args->i = i;
    
    while (true) {
      if (!thread_flags[t]) {
        thread_flags[t] = true;
        args->flag = &thread_flags[t];
        
        rc = pthread_create(&threads[t], NULL, MatrixMultLine, (void*)args);
        if (rc){
          printf("Error: unable to create thread, %d\n", rc);
          free(args);
          exit(0);
        }
        t = 0;
        break;
      }
      
      t++;
      if (t >= n_threads) {
        t = 0;
      }
    }
  }
  
  // Wait for all multiplication threads to complete
  for (int i = 0; i < n_threads; i++){
    if (thread_flags[i]) {
      rc = pthread_join(threads[i], NULL);
      if (rc){
        printf("Error: unable to join, %d\n", rc);
        exit(0);
      }
    }
  }

  return 1;
}
