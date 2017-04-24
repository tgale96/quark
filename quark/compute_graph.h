#ifndef QUARK_COMPUTE_GRAPH_H_
#define QUARK_COMPUTE_GRAPH_H_

#include "quark/common.h"
#include "quark/cuda_backend.h"
#include "quark/ops/op_base.h"
#include "quark/tensor.h"

namespace quark {

// Stores an operation and all of its parent/children operations
template <typename T>
struct Node {
  OpBase<T>* op = nullptr;
  vector<Node<T>*> parents;
  vector<Node<T>*> children;
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

  /**
   * @brief Prepares the graph for execution.
   *
   * Adds dependencies to the graph, checks for cycles, checks for in-place ops, and
   * constructs execution order for async graph execution.
   */
  void Compile();
  
  // Adds operation to the graph. If graph is compiled, calling this de-compiles the graph and
  // Compile() must be called again before execution.
  void AddOp(OpBase<T> *op);
  
private:
  vector<Node<T>> graph_;
};

} // namespace quark

#endif // QUARK_COMPUTE_GRAPH_H_
