cxx = g++
inc = ../utils/include
cflag = -g -Wall -O3 -I$(inc) -std=c++17 -fopenmp -O3
src_seq = futoshiki_openmp.cpp
src_seq += ../utils/src/puzzleio.cpp
src_seq += ../utils/src/puzzlegame.cpp
des_seq = futoshiki_openmp

all: $(des_seq)
.PHONY: clean

$(des_seq) : $(src_seq)
	$(cxx) $(cflag) $^ -o $@

clean:
	rm -f $(des_seq)