CXXFLAGS = -std=c++14 -g -v
CXXFLAGS += -I/usr/local/Cellar/eigen/3.4.0_1/include/
CXXFLAGS += -I/usr/local/Cellar/mpfr/4.1.0/include/
CXXFLAGS += -I/usr/local/Cellar/gmp/6.2.1_1/include/
CXXFLAGS += -I/usr/local/Cellar/libomp/14.0.6/include/
CXXFLAGS += -I/Users/theoares/libraries/mpreal/
CXXFLAGS += -L/usr/local/Cellar/mpfr/4.1.0/lib/
CXXFLAGS += -L/usr/local/Cellar/gmp/6.2.1_1/lib/
CXXFLAGS += -L/usr/local/Cellar/libmpc/1.2.1/lib

nevanlinna: nevanlinna.o
	clang++ -v -fverbose-asm nevanlinna.o -o Nevanlinna -lgmpxx -lmpfr -lgmp -lmpc

clean:
	rm -f *.o
	rm -f ./Nevanlinna
