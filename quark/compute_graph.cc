#include "quark/compute_graph.h"

namespace quark {

template <typename T>
void ComputeGraph<T>::Execute() {

}

template <typename T>
void ComputeGraph<T>::AddOp(OpBase *op, vector<Tensor<T, CudaBackend>> inputs,
    vector<Tensor<T, CudaBackend>> outputs) {
  // check for cycles
  // check for inplace ops
  // what other constraints are there on the graph?
}

} // namespace quark
