# Tools
NVCC     = nvcc
AR       = ar

# Flags
NVFLAGS  = -O2 -Iinclude
LDFLAGS  = -lcudart

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

# Compile C++ test files with nvcc
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(NVCC) $(NVFLAGS) -x cu -c $< -o $@

# Compile benchmark with nvcc
$(BENCHMARK_OBJ): $(BENCHMARK_SRC)
	$(NVCC) $(NVFLAGS) -x cu -c $< -o $@

# Archive static library
$(LIB_PATH): $(LIB_OBJ)
	$(AR) rcs $@ $^

# Link test binary
$(TEST_BIN): $(TEST_OBJ) $(LIB_PATH)
	$(NVCC) $(NVFLAGS) $^ $(LDFLAGS) -o $@

# Link benchmark binary
$(BENCHMARK_BIN): $(BENCHMARK_OBJ) $(LIB_PATH)
	$(NVCC) $(NVFLAGS) $^ $(LDFLAGS) -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)
