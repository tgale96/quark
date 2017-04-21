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

# gather list of cc files to build
CXX_FILES=$(shell find quark -name "*.cc")
CXX_OBJ=$(CXX_FILES:.cc=.o)

# handle header dependencies
DEP_FLAGS=-MMD -MP
DEPS=$(CXX_FILES:.cc=.d)

.PHONY: all clean

all: $(EXE)

%.o: %.cc
	@echo CXX $<
	$(Q)$(CC) $(CXX_FLAGS) $(DEP_FLAGS) $(INCLUDE) -c $< -o $@

$(EXE): $(CXX_OBJ)
	@echo CXX $@
	$(Q)$(CC) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(LIB)

clean: 
	$(Q)rm -f $(CXX_OBJ) $(DEPS) $(EXE)

-include $(DEPS)
