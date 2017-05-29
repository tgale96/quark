#include <gtest/gtest.h>

#include "quark/common.h"
#include "quark/compute_graph.h"
#include "quark/ops/math_ops.h"
#include "quark/ops/utility_ops.h"
#include "quark/test/quark_test.h"

namespace quark {

template <typename T>
class ComputeGraphTest : public QuarkTest<T> {
public:
  
protected:
};

typedef ::testing::Types <float, double> Implementations;
TYPED_TEST_CASE(ComputeGraphTest, Implementations);

// TODO(Trevor): Expand tests for the ComputeGraph
TYPED_TEST(ComputeGraphTest, TestBasicLine) {
  ComputeGraph<TypeParam> graph;
  vector<TypeParam> data = {1, 2, 3, 4};
  Tensor<TypeParam, CudaBackend> a({2, 2}, data);
  Tensor<TypeParam, CudaBackend> b;
  Tensor<TypeParam, CudaBackend> c;

  Copy(&graph, a, &b);
  Copy(&graph, b, &c);

  graph.Compile();
  graph.Execute();

  // check to make sure the data is the same
  ASSERT_TRUE(this->CompareData(a.data(), b.data(), a.size()));
  ASSERT_TRUE(this->CompareData(b.data(), c.data(), b.size()));
}

TYPED_TEST(ComputeGraphTest, TestBasicIndependent) {
  ComputeGraph<TypeParam> graph;
  vector<TypeParam> data = {1, 2, 3, 4};
  Tensor<TypeParam, CudaBackend> a({2, 2}, data);
  Tensor<TypeParam, CudaBackend> b;
  Tensor<TypeParam, CudaBackend> c;

  Copy(&graph, a, &b);
  Copy(&graph, a, &c);

  graph.Compile();
  graph.Execute();

  // check to make sure the data is the same
  ASSERT_TRUE(this->CompareData(a.data(), b.data(), a.size()));
  ASSERT_TRUE(this->CompareData(a.data(), c.data(), a.size()));
}

TYPED_TEST(ComputeGraphTest, TestBasicFork) {
  ComputeGraph<TypeParam> graph;
  vector<TypeParam> data = {1, 2, 3, 4};
  Tensor<TypeParam, CudaBackend> a({2, 2}, data);
  Tensor<TypeParam, CudaBackend> b;
  Tensor<TypeParam, CudaBackend> c;
  Tensor<TypeParam, CudaBackend> d;
  
  Copy(&graph, a, &b);
  Copy(&graph, b, &c);
  Copy(&graph, b, &d);
  
  graph.Compile();
  graph.Execute();

  // check to make sure the data is the same
  ASSERT_TRUE(this->CompareData(a.data(), b.data(), a.size()));
  ASSERT_TRUE(this->CompareData(b.data(), c.data(), b.size()));
  ASSERT_TRUE(this->CompareData(b.data(), d.data(), b.size()));
}

TYPED_TEST(ComputeGraphTest, TestBasicMerge) {
  ComputeGraph<TypeParam> graph;
  vector<TypeParam> data = {1, 2, 3, 4, 5, 6};
  Tensor<TypeParam, CudaBackend> a({2, 3}, data);
  Tensor<TypeParam, CudaBackend> b({2, 3}, data);

  vector<TypeParam> other_data = {6, 5, 4, 3, 2, 1};
  Tensor<TypeParam, CudaBackend> c({2, 3}, other_data);
  Tensor<TypeParam, CudaBackend> d({2, 3}, other_data);

  Tensor<TypeParam, CudaBackend> e, f, g;

  TypeParam constant = 1.0;
  
  Add(&graph, constant, false, a, constant, false, b, &e);
  Add(&graph, constant, false, c, constant, false, d, &f);
  Add(&graph, constant, false, e, constant, false, f, &g);

  graph.Compile();
  graph.Execute();

  
  // check to make sure the data is the same
  vector<TypeParam> res1 = {2, 4, 6, 8, 10, 12};
  vector<TypeParam> res2 = {12, 10, 8, 6, 4, 2};
  vector<TypeParam> res3 = {14, 14, 14, 14, 14, 14};
  ASSERT_TRUE(this->CompareData(e.data(), res1.data(), e.size()));
  ASSERT_TRUE(this->CompareData(f.data(), res2.data(), f.size()));
  ASSERT_TRUE(this->CompareData(g.data(), res3.data(), g.size()));
}

} // namespace quark
