cxx = g++
inc = ../utils/include
nvcc = nvcc
cuda_compile_flag = -arch sm_60 -rdc=true -I$(inc)
src = futoshiki_cuda.cu
src += ../utils/src/puzzlecuda.cu
des = futoshiki_cuda

.PHONY: clean
all: $(des)

$(des): $(src)
	$(nvcc) $(cuda_compile_flag) -o $@ $^

clean : 
	rm -f $(des)