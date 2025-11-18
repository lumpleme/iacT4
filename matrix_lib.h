#include <stdlib.h> 
#include <stdio.h>

struct matrix {
  unsigned long int height;
  unsigned long int width;
  float *rows;
};

//height = número de linhas da matriz (múltiplo de 8)
//width = número de colunas da matriz (múltiplo de 8)
//rows = sequência de linhas da matriz (height*width elementos)

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);
