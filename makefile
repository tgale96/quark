# Load the configuration file
CONFIG_FILE=makefile.config
ifeq ($(shell find . -name $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See README.md for instructions)
endif
include $(CONFIG_FILE)

# Basic flags and variables
PROJECT=quark
CXX_FLAGS=-std=c++11
DEP_FLAGS=-MMD -MP
NVCC=nvcc
SRC_DIR=src/$(PROJECT)
BUILD_DIR=build
EXAMPLE_DIR=examples
LIB_DIR=lib
STATIC_LIB=$(LIB_DIR)/lib$(PROJECT).a
Q=@

# Env variables common to all builds
INCLUDE=-Iinclude -I$(CUDA_DIR)/include
LDFLAGS=-L$(CUDA_DIR)/$(CUDA_LIB) -L$(LIB_DIR)
LIBS=-lcudart -lcublas -lquark

# Gather list of cc files to build
CXX_FILES=$(shell find $(SRC_DIR) -name "*.cc" ! -name "*_test.cc")
CXX_OBJ=$(CXX_FILES:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
LIB_DEPS=$(CXX_FILES:$(SRC_DIR)/%.cc=$(BUILD_DIR/%.d)

# Test env variables
TEST_INCLUDE=-I$(GTEST_DIR)/include $(INCLUDE)
TEST_LDFLAGS=-L$(GTEST_DIR) $(LDFLAGS)
TEST_LIB=-lgtest $(LIBS)
TEST_BIN=$(BUILD_DIR)/test/run_tests

# Gather list of test cc files
TEST_CXX_FILES=$(shell find $(SRC_DIR) -name "*_test.cc")
TEST_CXX_OBJ=$(TEST_CXX_FILES:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
TEST_DEPS=$(TEST_CXX_FILES:$(SRC_DIR/%.cc=$(BUILD_DIR)/%.d))

# Gather list of example cc files
EXAMPLE_CXX_FILES=$(shell find $(EXAMPLE_DIR) -name "*.cc")
EXAMPLE_BIN=$(EXAMPLE_CXX_FILES:%.cc=$(BUILD_DIR)/%)
EXAMPLE_DEPS=$(EXAMPLE_CXX_FILES:%.cc=$(BUILD_DIR)/%.d)

# Get directory structure
DIR_TREE=$(shell find $(SRC_DIR) -type d)
BUILD_DIR_TREE=$(DIR_TREE:$(SRC_DIR)%=$(BUILD_DIR)%)
BUILD_DIR_TREE+=$(BUILD_DIR)/$(EXAMPLE_DIR)

.PHONY: all clean test examples lib

all: lib test examples

$(BUILD_DIR_TREE):
	$(Q)mkdir -p $(BUILD_DIR_TREE)

# NOTE: Test build rules must be first so that correct *_test.o rule is called.
# This is not an issues with make version >=3.82
$(BUILD_DIR)/%_test.o: $(SRC_DIR)/%_test.cc | $(BUILD_DIR_TREE)
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(TEST_INCLUDE) -c $< -o $@

# Note: Having $(STATIC_LIB) as a dependency here is convenient, but it makes
# libquark.a be listed with the .o files when $(CC) is called. This doesn't
# seem to cause any issues, and I can't find any information on what happens
# when this occurs. Leaving this for now...
$(TEST_BIN): $(TEST_CXX_OBJ) $(STATIC_LIB) | $(BUILD_DIR_TREE)
	@echo CXX -o $@
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) -o $@ $^ $(TEST_LDFLAGS) $(TEST_LIB)

test: $(TEST_BIN)

$(BUILD_DIR)/$(EXAMPLE_DIR)/%: $(EXAMPLE_DIR)/%.cc $(STATIC_LIB) | $(BUILD_DIR_TREE)
	@echo CXX -o $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) -o $@ $< $(INCLUDE) $(LDFLAGS) $(LIBS)

examples: $(EXAMPLE_BIN)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR_TREE)
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(INCLUDE) -c $< -o $@

$(STATIC_LIB): $(CXX_OBJ)
	$(Q)mkdir -p $(LIB_DIR)
	@echo AR -o $@
	$(Q)ar rcs $@ $(CXX_OBJ)

lib: $(STATIC_LIB)

clean: 
	$(Q)rm -rf $(LIB_DIR) $(BUILD_DIR)

-include $(LIB_DEPS)
-include $(TEST_DEPS)
-include $(EXAMPLE_DEPS)
