#ifndef QUARK_COMPUTE_GRAPH_H_
#define QUARK_COMPUTE_GRAPH_H_

#include <queue>
#include <unordered_set>

#include "quark/common.h"
#include "quark/cuda_backend.h"
#include "quark/ops/op_base.h"
#include "quark/tensor.h"

namespace quark {

// Stores all parents and children of an op
struct EdgeList {
  std::unordered_set<OpId> parents;
  std::unordered_set<OpId> children;
};

// Stores everything needed to execute an op
template <typename T>
struct Pod {
  Pod(OpBase<T>* i_op, cudaStream_t i_stream, cudaEvent_t i_op_event, vector<cudaEvent_t> i_parent_op_events) :
    op(i_op), stream(i_stream), op_event(i_op_event), parent_op_events(i_parent_op_events) {}

  OpBase<T>* op;
  cudaStream_t stream;
  cudaEvent_t op_event;
  vector<cudaEvent_t> parent_op_events;
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
   * Runs all operations in the graph and blocks until completion.
   *
   * @throws runtime_error if the graph is empty
   */
  void Execute() {
    for (auto& pod : execution_pods_) {
      pod.op->Run(pod.stream, pod.op_event, pod.parent_op_events);
    }
    CUDA_CALL(cudaDeviceSynchronize());
  }

  /**
   * @brief Prepares the graph for execution.
   *
   * Adds dependencies to the graph, checks for cycles, checks for in-place ops, and
   * constructs execution order for async graph execution.
   *
   * Note: This may seem a bit weird that we wait to construct the graph until
   * it is compiled, but this is done for a reason. If we were to add edges to
   * the graph when the ops are added to the graph using "AddOp()", we would
   * need to have some way of checking whether the added op is a parent/child
   * for any existing ops. This gets complicated, as an op added at a later
   * time could be a parent of an already existing op (if the user add the
   * ops in reverse order, for example). To avoid the complexity of handling
   * these updates, we wait until we have knowledge of all the ops in the graph
   * to add the edges.
   */
  void Compile() {
    // Build a map of TensorIds to their parent operations OpId
    unordered_map<TensorId, OpId> tensor_parents;
    BuildTensorParentMap(&tensor_parents);

    // Connect the graph & gather the input nodes
    unordered_map<OpId, EdgeList> graph;
    std::queue<OpId> input_op_ids;
    BuildGraph(tensor_parents, &graph, &input_op_ids);

    // From the graph, build the vector of Pods to execute
    BuildExecutionOrder(&input_op_ids, &graph);
  }

  // Adds operation to the graph. If graph is compiled, calling this de-compiles the graph and
  // Compile() must be called again before execution.
  void AddOp(OpBase<T>* op) {
    QUARK_ASSERT(ops_.find(op->id()) == ops_.end(), "Operation already exists in the graph");

    // invalidate the compiled execution pods
    if (execution_pods_.size()) {
      execution_pods_.clear();
    }
    
    ops_[op->id()] = op;
  }
  
private:
  unordered_map<OpId, OpBase<T>*> ops_;
  vector<Pod<T>> execution_pods_;

  // Constructs map of tensors to their parent operations
  void BuildTensorParentMap(unordered_map<TensorId, OpId>* tensor_parents) {
    for (const auto& op_pair : ops_) {
      OpId current_op_id = op_pair.first;
      OpBase<T>* current_op = op_pair.second;
      const vector<GpuTensor<T>>& inputs = current_op->inputs();
      const vector<GpuTensor<T>>& outputs = current_op->outputs();
      
      for (const auto& output : outputs) {
        QUARK_CHECK(tensor_parents->find(output.id()) == tensor_parents->end(),
            "A tensor can only be the output of a single op");
        
        // mark this node as the parent of each of its output tensors
        (*tensor_parents)[output.id()] = current_op_id;

        // check for in-place operations
        for (const auto& input : inputs) {
          // TODO(Trevor): It would be nice if we could support in-place ops where
          // safe (would reduce memory footprint, add some usability for things
          // like parameter updates (+= op)).
          //
          // Note: It should be sufficient to just check the TensorIds, but just in-case the user
          // alters two tensors to point to the same chunk of memory we also check to make sure the
          // pointers are not equal. This will also fail in the case that both the tensors point
          // to nullptr, but this should be ok as ops will allocate Tensors as they are created.
          QUARK_CHECK(input.id() != output.id() && input.data() != output.data(), "In-place operations not supported");
        }
      }    
    }
  }
  
  // Adds all edges to the graph, collects the input nodes to the graph in the process
  void BuildGraph(const unordered_map<TensorId, OpId>& tensor_parents,
      unordered_map<OpId, EdgeList>* graph, std::queue<OpId>* input_op_ids) {    
    for (auto &op_pair : ops_) {
      OpId current_op_id = op_pair.first;
      OpBase<T>* current_op = op_pair.second;
      EdgeList* current_op_edges = &(*graph)[current_op_id];
      const vector<GpuTensor<T>>& inputs = current_op->inputs();
      const vector<GpuTensor<T>>& outputs = current_op->outputs();

      // add the ops parents and mark it as a child of its parents
      int num_graph_input = 0;
      for (const auto& input : inputs) {
        auto parent_node_it = tensor_parents.find(input.id());
        if (parent_node_it == tensor_parents.end()) {
          num_graph_input++;
        } else {
          // Note: if we insert the same key twice, the set will just ignore it. This is fine,
          // as the dependency between the two ops has already been registered
          OpId parent_op_id = parent_node_it->second;
          EdgeList* parent_op_edges = &(*graph)[parent_op_id];
          current_op_edges->parents.insert(parent_op_id);
          parent_op_edges->children.insert(current_op_id);
        }
      }

      // if all the ops inputs are graph inputs, this is an input node
      if (num_graph_input == inputs.size()) {
        input_op_ids->push(current_op_id);
      }
    }    
  }
  
  // Helper function to construct the order of execution for the graph
  void BuildExecutionOrder(std::queue<OpId> *input_op_ids, unordered_map<OpId, EdgeList>* graph) {
    /* while the queue of inputs is not emtpy:
     * 1. Pop the first node of it and add to the sorted list
     * 2. For each child of the node:
     *   a) Remove this node as a parent (find the best way to do this)
     *   b) If this node was the last parent, give no event, schedule in a 
     *      candidate stream; Then add this node to the list of inputs
     *   c) Else, add this nodes event to the childs list of events
     * Once queue is empty:
     * if any nodes have edges still: graph is not a DAG
     * store the execution order
     */
    unordered_map<OpId, vector<cudaEvent_t>> op_events;
    while (!input_op_ids->empty()) {
      OpId current_op_id = input_op_ids->front();
      OpBase<T>* current_op = ops_[current_op_id];
      EdgeList* current_op_edges = &(*graph)[current_op_id];
      input_op_ids->pop();

      // cudaEvent that signals the current ops completion; only enqueued if necessary
      cudaEvent_t event;
      bool event_allocated = false;
      
      for (const auto& child_op_id : current_op_edges->children) {
        EdgeList* child_op_edges = &(*graph)[child_op_id];
        
        // Remove the edge from the current node to this child
        current_op_edges->children.erase(child_op_id);
        child_op_edges->parents.erase(current_op_id);

        // if the node has no parents, add it to the input_nodes queue
        if (child_op_edges->parents.size() == 0) {
          input_op_ids->push(child_op_id);

          // TODO(Trevor): Add stream scheduling here
        } else {
          if (!event_allocated) {
            CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
          }
          
          // add this ops event to the child op's vector
          op_events[child_op_id].push_back(event);
        }
      }

      // TODO(Trevor): Add this ops stream insertion here
      // create the execution pod for the current op
      Pod<T> current_op_pod(current_op, 0, event, op_events[current_op_id]);
      execution_pods_.push_back(current_op_pod);
    }

    // If there are any edges remaining in the graph it is not acyclic
    for (const auto& node : *graph) {
      QUARK_CHECK(node.second.children.size() != 0, "Cannot compile, graph contains cycles");
    }
  }

  cudaStream_t AssignStream(unordered_map<OpId, cudaStream_t> op_stream_map) {
    /* For each parent to the current node (make sure we use a complete version of the graph):
     * 1. If parent has an available stream, take it; break out of loop
     * 2. If we still don't have stream at the end of the loop, allocate one (from StreamManager)
     * 3. Add stream to my list so that my children can do this process
     */
  }
  
  
};

} // namespace quark

#endif // QUARK_COMPUTE_GRAPH_H_
