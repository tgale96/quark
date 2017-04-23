#ifndef QUARK_COMPUTE_GRAPH_H_
#define QUARK_COMPUTE_GRAPH_H_

#include "quark/common.h"
#include "quark/cuda_backend.h"
#include "quark/ops/op_base.h"
#include "quark/tensor.h"

namespace quark {

// Stores an operation and all of its parent/children operations
struct Node {
  OpBase* op = nullptr;
  vector<OpBase*> parents;
  vector<OpBase*> children;
};

// TODO(Trevor): This object is templated on the type of data it works on, which
// is not ideal as it forces all ops in a graph to be done in the
// same precision. To get around this, the template parameter on 
// the Tensor object must be removed.
/**
 * @brief Graph composed of operations to be performed on GPU.
 *
 * A graph of operations that forms the core of Quark. A graph
 * is created by the user, and then ops are added to the graph.
 * The user executes the graph by calling the "Execute()" function.
 */
template <typename T>
class ComputeGraph {
public:
  /**
   * Constructs empty compute graph
   */
  ComputeGraph() {}

  DISABLE_COPY_ASSIGN_MOVE(ComputeGraph);
  
  ~ComputeGraph() = default;

  /**
   * Runs all operations in the graph.
   *
   * @throws runtime_error if the graph is empty
   * 
   * TODO(Trevor): add documentation on how the graph is executed
   * asynchronously.
   */
  void Execute();

  // Adds operation to the graph
  void AddOp(OpBase *op, vector<Tensor<T, CudaBackend>> inputs, vector<Tensor<T, CudaBackend>> outputs);
  
private:
  vector<Node> graph_;
  unordered_map<Tensor<T, CudaBackend>, Node> tensor_parents_;
};

} // namespace quark

#endif // QUARK_COMPUTE_GRAPH_H_
