# iacT4

gcc -o dat_creator dat_creator.c
 
./dat_creator 1024 1024 2.0 floats_1024_2.0f
 
./dat_creator 1024 1024 5.0 floats_1024_5.0f
 
./dat_creator 2048 2048 3.0 floats_2048_3.0f
 
./dat_creator 2048 2048 7.0 floats_2048_7.0f
 
gcc -pthread -mfma -mavx -Wall -o matrix_lib_test matrix_lib_test.c matrix_lib.c timer.c
 
./matrix_lib_test 4.0 1024 1024 1024 1024 floats_1024_2.0f.dat floats_1024_5.0f.dat result1.dat result2.dat
