# TODO(Trevor): The makefile currently outputs .o and .d files into the source
# directory. Move the build output to a build directory to reduce clutter
CC=g++-mp-4.9
CXX_FLAGS=-std=c++11
INCLUDE=-I. -I/Developer/NVIDIA/CUDA-7.5/include/
LDFLAGS=-L/Developer/NVIDIA/CUDA-7.5/lib
LIB=-lcudart
EXE=main
NVCC=nvcc
Q=@

# Test vars
TEST_INCLUDE=-I//Users/trevorgale/googletest-release-1.8.0/googletest/include $(INCLUDE)
TEST_LDFLAGS=-L/Users/trevorgale/googletest-release-1.8.0/googletest $(LDFLAGS)
TEST_LIB=-lgtest $(LIB)
TEST_EXE=quark/test/run_tests

# Gather list of cc files to build
CXX_FILES=$(shell find quark -name "*.cc" ! -name "*_test.cc")
CXX_OBJ=$(CXX_FILES:.cc=.o)

# Handle header dependencies
DEP_FLAGS=-MMD -MP
DEPS=$(CXX_FILES:.cc=.d)

# Gather list of test cc files
TEST_CXX_FILES=$(shell find quark -name "*_test.cc")
TEST_CXX_FILES+=$(shell find quark -name "*.cc" ! -name "main.cc")
TEST_CXX_OBJ=$(TEST_CXX_FILES:.cc=.o)
TEST_DEPS=$(TEST_CXX_FILES:.cc=.d)

.PHONY: all clean

all: $(EXE)

# NOTE: Test build rules must be first so that correct *_test.o rule is called.
# This is not an issues with make version >=3.82
%_test.o: %_test.cc
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(TEST_INCLUDE) -c $< -o $@

test: $(TEST_CXX_OBJ)
	@echo CXX $(TEST_EXE)
	$(Q)$(CC) $(CXX_FLAGS) -o $(TEST_EXE) $^ $(TEST_LDFLAGS) $(TEST_LIB)

%.o: %.cc
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(INCLUDE) -c $< -o $@

$(EXE): $(CXX_OBJ)
	@echo CXX $@
	$(Q)$(CC) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(LIB)

clean: 
	$(Q)rm -f $(CXX_OBJ) $(DEPS) $(EXE) $(TEST_CXX_OBJ) $(TEST_DEPS)

-include $(DEPS)
-include $(TEST_DEPS)
