NDEBUG := 0

ifeq ($(NDEBUG), 1)
	NDEBUGOP = -DNDEBUG
else
	NDEBUGOP =
endif

DEBUG := 0
ifeq ($(DEBUG), 1)
	DEBUGOP = -DDEBUG -pg -ggdb -O2
	DEBUGLP = -pg -O2
else
	DEBUGOP = -funsafe-math-optimizations -Ofast -flto=auto  -funroll-all-loops -pipe -march=native $(NDEBUGOP)
	DEBUGLP = -Ofast
endif

# bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109569
EXTRAOP = -Wno-array-bounds -Wno-stringop-overflow

CUDAOP = -DCUDA
CUDAIC = -I/opt/cuda/include
CUDALP = -L/opt/cuda/lib64 -lcudart -lcublas -lcudnn
INC3RD = -Iinclude/utility/./3rdparty/spdlog-1.11.0/include

#LOP       = -Wl,--gc-sections -flto -fopt-info-vec-optimized $(DEBUGLP) $(CUDALP)
LOP       = -Wl,--gc-sections -flto $(DEBUGLP) $(CUDALP)
OP        = -fconcepts-diagnostics-depth=16 -ftemplate-depth=1024 $(DEBUGOP) $(CUDAOP) $(CUDAIC) $(INC3RD)

CXX       = g++
CXXFLAGS  = -std=c++20 -Wall -Wpedantic -Wextra -fmax-errors=2 -Werror -ftemplate-backtrace-limit=0 $(OP) $(EXTRAOP)
LFLAGS    = -pthread -lstdc++fs ${LOP}
LINK      = $(CXX)
NVCC      = nvcc
NVCCFLAGS = -ccbin $(CXX) -m64 --threads 0 --std=c++20 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = ./lib
LOG_DIR       = .

all: test

#src/cuda/nodes/vocabulary_projection.cpp
vocabulary_projection.o: src/cuda/nodes/vocabulary_projection.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/vocabulary_projection.o src/cuda/nodes/vocabulary_projection.cpp

device_management.o: src/cuda/device_management.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/device_management.o src/cuda/device_management.cpp

memory_management.o: src/cuda/memory_management.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/memory_management.o src/cuda/memory_management.cpp

node.o: src/cuda/node.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/node.o src/cuda/node.cpp

libnnl_cuda.a: memory_management.o node.o device_management.o
	ar rcs $(LIB_DIR)/libnnl_cuda.a $(OBJECTS_DIR)/memory_management.o $(OBJECTS_DIR)/node.o $(OBJECTS_DIR)/device_management.o

kernel_cuda_add_bias.o: src/cuda/kernels/add_bias.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/add_bias.cu -o $(OBJECTS_DIR)/kernel_cuda_add_bias.o

kernel_cuda_gelu.o: src/cuda/kernels/cuda_gelu.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_gelu.cu -o $(OBJECTS_DIR)/kernel_cuda_gelu.o

kernel_cuda_add.o: src/cuda/kernels/cuda_add.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_add.cu -o $(OBJECTS_DIR)/kernel_cuda_add.o

kernel_cuda_layer_norm.o: src/cuda/kernels/cuda_layer_norm.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_layer_norm.cu -o $(OBJECTS_DIR)/kernel_cuda_layer_norm.o

kernel_cuda_scaled_offset.o: src/cuda/kernels/cuda_scaled_offset.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_scaled_offset.cu -o $(OBJECTS_DIR)/kernel_cuda_scaled_offset.o

kernel_cuda_scaled_mask.o: src/cuda/kernels/cuda_scaled_mask.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_scaled_mask.cu -o $(OBJECTS_DIR)/kernel_cuda_scaled_mask.o

kernel_cuda_argmax_1d.o: src/cuda/kernels/cuda_argmax_1d.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_argmax_1d.cu -o $(OBJECTS_DIR)/kernel_cuda_argmax_1d.o

kernel_cuda_vocabulary_lookup.o: src/cuda/kernels/cuda_vocabulary_lookup.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels/cuda_vocabulary_lookup.cu -o $(OBJECTS_DIR)/kernel_cuda_vocabulary_lookup.o

libnnl_cuda_kernels.a: kernel_cuda_add_bias.o kernel_cuda_gelu.o kernel_cuda_scaled_offset.o kernel_cuda_scaled_offset.o kernel_cuda_scaled_mask.o kernel_cuda_layer_norm.o kernel_cuda_add.o vocabulary_projection.o kernel_cuda_argmax_1d.o kernel_cuda_vocabulary_lookup.o
	ar rcs $(LIB_DIR)/libnnl_cuda_kernels.a $(OBJECTS_DIR)/kernel_cuda_add_bias.o $(OBJECTS_DIR)/kernel_cuda_gelu.o $(OBJECTS_DIR)/kernel_cuda_layer_norm.o $(OBJECTS_DIR)/kernel_cuda_scaled_offset.o $(OBJECTS_DIR)/kernel_cuda_scaled_mask.o $(OBJECTS_DIR)/kernel_cuda_add.o $(OBJECTS_DIR)/vocabulary_projection.o $(OBJECTS_DIR)/kernel_cuda_argmax_1d.o $(OBJECTS_DIR)/kernel_cuda_vocabulary_lookup.o

tests.o: tests/tests.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/tests.o tests/tests.cc

test: tests.o libnnl_cuda.a libnnl_cuda_kernels.a
	$(LINK) -o $(BIN_DIR)/tests $(OBJECTS_DIR)/tests.o $(LIB_DIR)/libnnl_cuda.a $(LIB_DIR)/libnnl_cuda_kernels.a $(LFLAGS)

gpt2_tokenizer.o:
	$(CXX) $(CXXFLAGS) -c ./examples/gpt2-1558M/test_gpt2_tokenizer.cc -o $(OBJECTS_DIR)/gpt2_tokenizer.o

test_gpt2_tokenizer: gpt2_tokenizer.o simdjson.o
	$(LINK) -o $(BIN_DIR)/tests_gpt2_tokenizer $(OBJECTS_DIR)/gpt2_tokenizer.o $(OBJECTS_DIR)/simdjson.o $(LFLAGS)

gpt2_1558m.o: examples/gpt2-1558M/main.cc
	$(CXX) $(CXXFLAGS) -c ./examples/gpt2-1558M/main.cc -o $(OBJECTS_DIR)/gpt2_1558m.o

simdjson.o: examples/gpt2-1558M/simdjson.cpp
	$(CXX) $(CXXFLAGS) -c ./examples/gpt2-1558M/simdjson.cpp -o $(OBJECTS_DIR)/simdjson.o

gpt2_1558m: gpt2_1558m.o simdjson.o libnnl_cuda.a libnnl_cuda_kernels.a
	$(LINK) -o $(BIN_DIR)/gpt2_1558m $(OBJECTS_DIR)/gpt2_1558m.o $(OBJECTS_DIR)/simdjson.o $(LIB_DIR)/libnnl_cuda.a $(LIB_DIR)/libnnl_cuda_kernels.a $(LFLAGS)


clean:
	rm -rf ./obj/*.o
	rm -rf ./lib/*.a
	rm -rf ./*.txt
	rm -rf ./*.bin
	rm -rf ./*.log
	rm -rf ./*.out


