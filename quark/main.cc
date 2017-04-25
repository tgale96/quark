#include <iostream>

#include "quark/compute_graph.h"
#include "quark/cpu_backend.h"
#include "quark/cuda_backend.h"
#include "quark/tensor.h"

int main() {
  quark::ComputeGraph<float> graph;
  graph.Compile();
  quark::Tensor<float, quark::CpuBackend> test;
  std::cout << "This is a test" << std::endl;
}
