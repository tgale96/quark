#include <iostream>

#include "quark/compute_graph.h"
#include "quark/cpu_backend.h"
#include "quark/cuda_backend.h"
#include "quark/ops/utility_ops.h"
#include "quark/tensor.h"
using namespace std;
using namespace quark;

int main() {
  ComputeGraph<float> graph;
  vector<float> data = {1, 2, 3, 4};
  Tensor<float, CudaBackend> a({2, 2}, data);
  Tensor<float, CudaBackend> b;

  Copy(&graph, a, &b);
  graph.Compile();
}
