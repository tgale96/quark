#include "quark/compute_graph.h"

namespace quark {

// template <typename T>
// void ComputeGraph<T>::Execute() {

// }

// Note: This may seem a bit weird that we wait to construct the graph until
// it is compiled, but this is done for a reason. If we were to add edges to
// the graph when the ops are added to the graph using "AddOp()", we would
// need to have some way of checking whether the added op is a parent/child
// for any existing ops. This gets complicated, as an op added at a later
// time could be a parent of an already existing op (if the user add the
// ops in reverse order, for example). To avoid the complexity of handling
// these updates, we wait until we have knowledge of all the ops in the graph
// to add the edges.
// template <typename T>
// void ComputeGraph<T>::Compile() {
//   unordered_map<const GpuTensor<T>&, Node<T>*> tensor_parents;  
//   for (const auto& node : graph_) {
//     OpBase<T>* op = node.op;
//     vector<const GpuTensor<T>&> inputs = op->inputs();
//     vector<const GpuTensor<T>&> outputs = op->outputs();

//     for (const auto& output : outputs) {
//       QUARK_CHECK(tensor_parents.find(output) == tensor_parents.end(), "A tensor can only be the output of a single op");
//       // mark this node as the parent of each of its output tensors
//       tensor_parents[output] = &node;

//       // check for in-place operations
//       for (const auto& input : inputs) {
//         // TODO(Trevor): It would be nice if we could support in-place ops where safe (would reduce
//         // memory footprint, add some usability for things like parameter updates (+= op).
//         QUARK_CHECK(input != output, "In-place operations not supported");
//       }
//     }    
//   }

//   // connect the graph
//   vector<Node<T>*> input_nodes;
//   for (auto &node : graph_) {
//     OpBase<T>* op = node.op;
//     vector<const GpuTensor<T>&> inputs = op->inputs();
//     vector<const GpuTensor<T>&> outputs = op->outputs();

//     // add the ops parents and mark it as a child of its parents
//     int num_graph_input = 0;
//     for (const auto& input : inputs) {
//       auto parent_node_it = tensor_parents.find(input);
//       if (parent_node_it == tensor_parents.end()) {
//         // this tensor is an input tensor
//         num_graph_input++;
//       } else {
//         node.parents.append(parent_node_it->second);
//         parent_node_it->second->children.append(node);
//       }
//     }

//     // if all the ops inputs are graph inputs, this is an input node
//     if (num_graph_input == inputs.size()) {
//       input_nodes.append(&node);
//     }
//   }

//   // check the graph for cycles

//   // build execution order
// }

// template <typename T>
// void ComputeGraph<T>::AddOp(OpBase<T> *op) {
//   Node<T> new_node;
//   new_node.op = op;
//   graph_.append(new_node);
// }

} // namespace quark
