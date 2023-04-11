CXXFLAGS = -std=c++14 -g -v
# CXXFLAGS += -L$(brew --prefix)/include
# -I flag tells compiler to look in these directories.
CXXFLAGS += -I/Users/willjay/GitHub/mpreal/
CXXFLAGS += -I/opt/homebrew/Cellar/mpfr/4.2.0/include/
CXXFLAGS += -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/
CXXFLAGS += -I/opt/homebrew/Cellar/gmp/6.2.1_1/include/
CXXFLAGS += -I/opt/homebrew/Cellar/libomp/14.0.6/include/
CXXFLAGS += -L/opt/homebrew/Cellar/mpfr/4.2.0/lib
CXXFLAGS += -L/opt/homebrew/Cellar/gmp/6.2.1_1/lib

# CXXFLAGS += -I/usr/local/Cellar/eigen/3.4.0_1/include/
# CXXFLAGS += -I/usr/local/Cellar/mpfr/4.1.0/include/
# CXXFLAGS += -I/usr/local/Cellar/gmp/6.2.1_1/include/
# CXXFLAGS += -I/usr/local/Cellar/libomp/14.0.6/include/
# CXXFLAGS += -I/Users/theoares/mpreal/
# CXXFLAGS += -L$(brew --prefix)/lib
# CXXFLAGS += -L/usr/local/Cellar/mpfr/4.1.0/lib/
# CXXFLAGS += -L/usr/local/Cellar/gmp/6.2.1_1/lib/

#CXXFLAGS=-std=c++14 -g -I/usr/local/include/eigen3/ -I/Users/wijay/GitHub/mpreal/

hlt: HLTClass.o
	clang++ -v -fverbose-asm HLTClass.o -o HLTClass -lmpfr -lgmp
	# clang++ -lmpfr -lgmp -std=c++11 -o HLTClass HLTClass.cpp

clean:
	rm -f *.o
