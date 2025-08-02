# Tools
NVCC     = nvcc
AR       = ar
GPP      = g++

# Configuration
BUILD_TYPE ?= release

# Directories
SRC_DIR      = src
INCLUDE_DIR  = include
TEST_DIR     = tests
BENCH_DIR    = benchmarks
BUILD_DIR    = build/$(BUILD_TYPE)

# Library
LIB_NAME     = libbitonic_sort.a
LIB_PATH     = $(BUILD_DIR)/$(LIB_NAME)
LIB_SRC      = $(wildcard $(SRC_DIR)/*.cu)
LIB_OBJ      = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(LIB_SRC))

# Tests
TEST_SRC     = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ     = $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(TEST_SRC))
TEST_BIN     = $(BUILD_DIR)/tests

# Benchmarks
BENCHMARK_SRC  = $(BENCH_DIR)/benchmark.cpp
BENCHMARK_OBJ  = $(BUILD_DIR)/benchmark.o
BENCHMARK_BIN  = $(BUILD_DIR)/benchmarks

# Flags
COMMON_FLAGS      = -I$(INCLUDE_DIR)
NVFLAGS_DEBUG     = -g -G -O0 -DDEBUG_BUILD $(COMMON_FLAGS)
NVFLAGS_RELEASE   = -O2 $(COMMON_FLAGS)
CPPFLAGS_DEBUG    = -std=c++11 -g -O0 -DDEBUG_BUILD $(COMMON_FLAGS) -I$(CUDA_HOME)/include
CPPFLAGS_RELEASE  = -std=c++11 -O2 $(COMMON_FLAGS) -I$(CUDA_HOME)/include
LDFLAGS           = -L$(CUDA_HOME)/lib64 -lcudart

# Select flags based on build type
ifeq ($(BUILD_TYPE),debug)
    NVFLAGS   = $(NVFLAGS_DEBUG)
    CPPFLAGS  = $(CPPFLAGS_DEBUG)
else
    NVFLAGS   = $(NVFLAGS_RELEASE)
    CPPFLAGS  = $(CPPFLAGS_RELEASE)
endif

# Default target
all: $(BUILD_DIR) $(LIB_PATH) $(TEST_BIN) $(BENCHMARK_BIN)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA source files with nvcc
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Compile test cpp files with g++
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(GPP) $(CPPFLAGS) -c $< -o $@

# Compile benchmark cpp with g++
$(BUILD_DIR)/benchmark.o: $(BENCHMARK_SRC)
	$(GPP) $(CPPFLAGS) -c $< -o $@

# Archive static library
$(LIB_PATH): $(LIB_OBJ)
	$(AR) rcs $@ $^

# Link test binary with g++
$(TEST_BIN): $(TEST_OBJ) $(LIB_PATH)
	$(GPP) $^ $(LDFLAGS) -o $@

# Link benchmark binary with g++
$(BENCHMARK_BIN): $(BENCHMARK_OBJ) $(LIB_PATH)
	$(GPP) $^ $(LDFLAGS) -o $@

# Clean
clean:
	rm -rf build
