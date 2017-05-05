#ifndef QUARK_TEST_QUARK_TEST_H_
#define QUARK_TEST_QUARK_TEST_H_

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <gtest/gtest.h>
#include <random>

namespace quark {

template <typename T>
class QuarkTest : public ::testing::Test {
public:
  void SetUp() {
    std::srand(std::time(0));
  }

  vector<int64> GetRandDims() {
    int dims = std::rand() % 10 + 1;

    vector<int64> shape;
    for (int i = 0; i < dims; ++i) {
      shape.push_back(int64(std::rand() % 5 + 1));
    }

    return shape;
  }

  void GetRandData(vector<int64> dims, vector<T> *data) {
    QUARK_CHECK(data != nullptr, "Input pointer must not be nullptr");
    int64 num = Prod(dims);
    data->resize(num);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distr;
    
    for (int i = 0; i < num; ++i) {
      (*data)[i] = (T)distr(gen);
    }
  }

  bool CompareData(const T* d1, const T* d2, int64 num, const double threshold = 0.0) {
    // check the data
    T* h_d1 = new T[num];
    T* h_d2 = new T[num];
    CUDA_CALL(cudaMemcpy(h_d1, d1, num * sizeof(T), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(h_d2, d2, num * sizeof(T), cudaMemcpyDefault));

    bool res = true;
    for (int i = 0; i < num; ++i) {
      double diff = abs(h_d1[i] - h_d2[i]);
      if (diff > threshold) {
        std::cout << h_d1[i] << " v. " << h_d2[i] << std::endl;
        res = false;
      }
    }
    
    delete[] h_d1;
    delete[] h_d2;
    return res;
  }
  
protected:
};
  
} // namespace quark

#endif // QUARK_TEST_QUARK_TEST_H_
