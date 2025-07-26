# Tools
NVCC     = nvcc
CXX      = g++
AR       = ar

# Flags
CUDA_INC ?= /usr/include
CUDA_LIB ?= /usr/lib/x86_64-linux-gnu

CXXFLAGS = -O2 -Wall -I$(CUDA_INC) -Iinclude
NVFLAGS  = -O2 -I$(CUDA_INC) -Iinclude
LDFLAGS  = -L$(CUDA_LIB) -lcudart

# File structure
SRC_DIR     = src
BUILD_DIR   = build
INCLUDE_DIR = include
TEST_DIR    = tests
BENCH_DIR   = benchmarks

# Library setup
LIB_NAME    = libbitonic_sort.a
LIB_PATH    = $(BUILD_DIR)/$(LIB_NAME)
LIB_SRC     = $(wildcard $(SRC_DIR)/*.cu)
LIB_OBJ     = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(LIB_SRC))

# Test setup
TEST_SRC    = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ    = $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(TEST_SRC))
TEST_BIN    = $(BUILD_DIR)/tests

# Benchmark setup
BENCHMARK_SRC  = $(BENCH_DIR)/benchmark.cpp
BENCHMARK_OBJ  = $(BUILD_DIR)/benchmark.o
BENCHMARK_BIN  = $(BUILD_DIR)/benchmark

# Default target
all: $(BUILD_DIR) $(LIB_PATH) $(TEST_BIN) $(BENCHMARK_BIN)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Archive static library in build directory
$(LIB_PATH): $(LIB_OBJ)
	$(AR) rcs $@ $^

# Compile C++ test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile benchmark source
$(BENCHMARK_OBJ): $(BENCHMARK_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link test binary
$(TEST_BIN): $(TEST_OBJ) $(LIB_PATH)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Link benchmark binary
$(BENCHMARK_BIN): $(BENCHMARK_OBJ) $(LIB_PATH)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)
