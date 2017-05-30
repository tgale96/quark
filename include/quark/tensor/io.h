#ifndef QUARK_TENSOR_IO_H_
#define QUARK_TENSOR_IO_H_

#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "quark/common.h"
#include "quark/tensor/tensor.h"
#include "quark/util/backend_util.h"

namespace quark {

/**
 * @Loads a data from a text file into a Tensor. Each line of the input file is loaded as a row.
 *
 * This function expects the data to be stored as CSV, where each line represents a row in a
 * matrix. Each line must contain the same number of data points.
 */
template <typename T, typename Backend>
void LoadFromTextFile(const string file_name, Tensor<T, Backend>* t) {
  std::ifstream f(file_name);
  vector<T> data;

  {
    string first_line;
    std::getline(f, first_line);
    std::istringstream iss(first_line);
    
    for (std::string tmp; getline(iss, tmp, ',');) {
      T val = strtof(tmp.c_str(), nullptr);
      data.push_back(val);
    }
  }
    
  int64 cols = data.size();
  QUARK_CHECK(cols > 0, "Could not read any values from the first line of " + file_name);

  // read in the remaining lines
  for (std::string line; getline(f, line);) {
    std::istringstream iss(line);
    int64 line_size = 0;
    for (std::string str_val; getline(iss, str_val, ',');) {
      T val = strtof(str_val.c_str(), nullptr);
      data.push_back(val);
      line_size++;
    }
    QUARK_CHECK(line_size == cols,
        "All lines in input file must contain the same number of data points");
  }

  // load the data into the tensor
  vector<int64> dims = {int64(data.size()) / cols, cols};
  t->Resize(dims);

  CopyData(data.size(), data.data(), t->mutable_data());
}

/**
 * @brief Fills the input Tensor with random values (between 0 and 1)
 */
template <typename T, typename Backend>
void InitRandomTensor(vector<int64> shape, Tensor<T, Backend>* t) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distr;
  vector<T> data(Prod(shape), 0.0);

  t->Resize(shape);  
  for (auto& val : data) val = (T)distr(gen);
  CopyData(data.size(), data.data(), t->mutable_data());
}

/**
 * @brief Writes Tensor to input file. Each row (or outer most dimension) is places on a new line
 */
template <typename T, typename Backend>
void WriteToTextFile(string file_name, const Tensor<T, Backend>& t) {
  std::ofstream f(file_name);
  Tensor<T, CpuBackend> h_t(t.shape());
  h_t.Copy(t);

  int64 outer_dim = h_t.shape()[0];
  int64 inner_dim = Prod(h_t.shape()) / outer_dim;

  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      f << h_t.data()[i * inner_dim + j] << " ";
    }
    f << std::endl;
  }  
}

} // namespace quark

#endif // QUARK_TENSOR_IO_H_
