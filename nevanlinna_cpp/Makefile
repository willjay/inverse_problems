CXXFLAGS = -std=c++14 -g -v
# CXXFLAGS += -I/usr/local/Cellar/eigen/3.4.0_1/include/
# CXXFLAGS += -I/usr/local/Cellar/mpfr/4.1.0-p13/include/
# CXXFLAGS += -I/usr/local/Cellar/gmp/6.2.1_1/include/
# CXXFLAGS += -I/usr/local/Cellar/libomp/15.0.6/include/
# CXXFLAGS += -I/Users/theoares/libraries/mpreal/
# CXXFLAGS += -L/usr/local/Cellar/mpfr/4.1.0-p13/lib/
# CXXFLAGS += -L/usr/local/Cellar/gmp/6.2.1_1/lib/

CXXFLAGS += -I/Users/willjay/GitHub/mpreal
CXXFLAGS += -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/
CXXFLAGS += -I/opt/homebrew/Cellar/mpfr/4.2.0/include/
CXXFLAGS += -I/opt/homebrew/Cellar/gmp/6.2.1_1/include/
CXXFLAGS += -I/opt/homebrew/Cellar/libomp/14.0.6/include/
CXXFLAGS += -I/opt/homebrew/Cellar/hdf5/1.12.2_2/include/
CXXFLAGS += -L/opt/homebrew/Cellar/mpfr/4.2.0/lib
CXXFLAGS += -L/opt/homebrew/Cellar/gmp/6.2.1_1/lib
CXXFLAGS += -L/opt/homebrew/Cellar/hdf5/1.12.2_2/lib

nevanlinna: nevanlinna.o
	clang++ -v -fverbose-asm nevanlinna.o -o Nevanlinna -lgmpxx -lmpfr -lgmp -lhdf5 -lhdf5_cpp

clean:
	rm -f *.o
	rm -f ./Nevanlinna
