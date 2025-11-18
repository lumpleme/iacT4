#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  
  if (argc != 5) {
    printf("Usage: %s <matrix_height> <matrix_width> <float_value> <filename>\n", argv[0]);
    return 0;
  }
  
  unsigned long int m = strtoul(argv[1], NULL, 10);
  unsigned long int n = strtoul(argv[2], NULL, 10);
  float value = strtof(argv[3], NULL);
  char *filename = argv[4];
  char *dat = ".dat";
  
  strcat(filename, dat);
  
  FILE *file = fopen(filename, "wb");
  if (file == NULL) {
    printf("Error: Could not create file %s\n", filename);
    return 1;
  }
  
  for (unsigned long int i = 0; i < m; i++) {
    for (unsigned long int j = 0; j < n; j++) {
      fwrite(&value, sizeof(float), 1, file);
    }
  }
  
  fclose(file);
  
  printf("Data file '%s' created successfully.\n", filename);
  
  return 0;
}
