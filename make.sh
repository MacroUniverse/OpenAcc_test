# g++ -fopenacc -fopt-info-all test1.cpp -o test1.x # doesn't work
nvc++ -acc -gpu=managed -Minfo=accel test1.cpp -o test1.x
