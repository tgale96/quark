# Load the configuration file
CONFIG_FILE=makefile.config
ifeq ($(shell find . -name $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See README.md for instructions)
endif
include $(CONFIG_FILE)

# Basic flags and variables
CXX_FLAGS=-std=c++11
DEP_FLAGS=-MMD -MP
NVCC=nvcc
SRC_DIR=quark
BUILD_DIR=build
Q=@

# Lib env variables
INCLUDE=-I. -I$(CUDA_DIR)/include
LDFLAGS=-L$(CUDA_DIR)/$(CUDA_LIB)
LIB=-lcudart -lcublas
EXE=main

# Gather list of cc files to build
CXX_FILES=$(shell find $(SRC_DIR) -name "*.cc" ! -name "*_test.cc")
CXX_OBJ=$(CXX_FILES:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
DEPS=$(CXX_FILES:$(SRC_DIR)/%.cc=$(BUILD_DIR/%.d)

# Test env variables
TEST_INCLUDE=-I$(GTEST_DIR)/include $(INCLUDE)
TEST_LDFLAGS=-L$(GTEST_DIR) $(LDFLAGS)
TEST_LIB=-lgtest $(LIB)
TEST_EXE=$(BUILD_DIR)/test/run_tests
TEST_EXE_LINK=run_tests

# TODO(Trevor): Clean this up once we move to static lib build
# Gather list of test cc files
TEST_CXX_FILES=$(shell find $(SRC_DIR) -name "*_test.cc")
TEST_CXX_FILES+=$(shell find $(SRC_DIR) -name "*.cc" ! -name "main.cc")
TEST_CXX_OBJ=$(TEST_CXX_FILES:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
TEST_DEPS=$(TEST_CXX_FILES:$(SRC_DIR/%.cc=$(BUILD_DIR)/%.d))

# Get directory structure
DIR_TREE=$(shell find $(SRC_DIR) -type d)
BUILD_DIR_TREE=$(DIR_TREE:$(SRC_DIR)%=$(BUILD_DIR)%)

.PHONY: all clean test

all: $(EXE)

$(BUILD_DIR_TREE):
	$(Q)mkdir -p $(BUILD_DIR_TREE)

# NOTE: Test build rules must be first so that correct *_test.o rule is called.
# This is not an issues with make version >=3.82
$(BUILD_DIR)/%_test.o: $(SRC_DIR)/%_test.cc | $(BUILD_DIR_TREE)
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(TEST_INCLUDE) -c $< -o $@

test: $(TEST_CXX_OBJ)
	@echo CXX $(TEST_EXE)
	$(Q)$(CC) $(CXX_FLAGS) -o $(TEST_EXE) $^ $(TEST_LDFLAGS) $(TEST_LIB)
	$(Q)ln -s $(TEST_EXE) $(TEST_EXE_LINK)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR_TREE)
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(INCLUDE) -c $< -o $@

$(EXE): $(CXX_OBJ)
	@echo CXX $@
	$(Q)$(CC) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(LIB)

clean: 
	$(Q)rm -rf $(EXE) $(TEST_EXE_LINK) $(BUILD_DIR)

-include $(DEPS)
-include $(TEST_DEPS)
