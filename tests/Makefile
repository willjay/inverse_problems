CXXFLAGS = -std=c++14 -g -v
CXXFLAGS += -I/usr/local/Cellar/eigen/3.4.0_1/include/
CXXFLAGS += -I/usr/local/Cellar/mpfr/4.1.0-p13/include/
CXXFLAGS += -I/usr/local/Cellar/gmp/6.2.1_1/include/
CXXFLAGS += -I/usr/local/Cellar/libomp/15.0.6/include/
CXXFLAGS += -I/Users/theoares/libraries/mpreal/
CXXFLAGS += -L/usr/local/Cellar/mpfr/4.1.0-p13/lib/
CXXFLAGS += -L/usr/local/Cellar/gmp/6.2.1_1/lib/

tests: tests.o
	clang++ -v -fverbose-asm tests.o -o Tests -lgmpxx -lmpfr -lgmp

clean:
	rm -f *.o
	rm -f ./Tests
