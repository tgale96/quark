#include <chrono>
#include <iostream>

#include "quark/compute_graph.h"
#include "quark/io.h"
#include "quark/ops/math_ops.h"
#include "quark/ops/utility_ops.h"
#include "quark/tensor.h"

using namespace std;
using namespace quark;

int main() {
  // Load the data and labels & initialize beta
  Tensor<float, CudaBackend> data, labels, beta;
  LoadFromTextFile("./dummy_data/small_train.data", &data);
  LoadFromTextFile("./dummy_data/small_train.labels", &labels);
  InitRandomTensor({data.shape()[1], 1}, &beta);

  int num_samples = data.shape()[0];
  int max_iter = 100;
  float learning_rate = .01;
  
  // Intermediate results
  Tensor<float, CudaBackend> xb, y_m_xb, loss, grad, new_beta;

  // Build the graph
  ComputeGraph<float> cg;

  Matmul(&cg, 1.0f, false, data, false, beta, &xb);

  Add(&cg, 1.0f, false, labels, -1.0f, false, xb, &y_m_xb);

  Matmul(&cg, 1.0f / num_samples, true, y_m_xb, false, y_m_xb, &loss);

  Matmul(&cg, -2.0f / num_samples, true, data, false, y_m_xb, &grad);

  Add(&cg, 1.0f, false, beta, -learning_rate, false, grad, &new_beta);
      
  cg.Compile();

  chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
  for (int i = 0; i < max_iter; ++i) {    
    cg.Execute();
    beta.Copy(new_beta);

    chrono::duration<double> run_time = chrono::system_clock::now() - start_time;
    cout << "[elapsed_time: " << run_time.count() << "] iter " << i << ", regularized_mse " << loss;
  }

  // Write beta to file
  WriteToTextFile("beta", beta);

  // Compute MSE on test set
  Tensor<float, CudaBackend> test_data, test_labels;
  LoadFromTextFile("./dummy_data/small_test.data", &test_data);
  LoadFromTextFile("./dummy_data/small_test.labels", &test_labels);

  int num_test_samples = test_data.shape()[0];
  ComputeGraph<float> tg;

  Matmul(&tg, 1.0f, false, test_data, false, beta, &xb);
  Add(&tg, 1.0f, false, test_labels, -1.0f, false, xb, &y_m_xb);
  Matmul(&tg, 1.0f / num_test_samples, true, y_m_xb, false, y_m_xb, &loss);
  tg.Compile();
  tg.Execute();
  cout << "Test MSE: " << loss;
}
