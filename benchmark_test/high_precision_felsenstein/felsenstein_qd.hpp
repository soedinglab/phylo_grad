#pragma once

#include <qd/qd_real.h>

#include <Eigen/Dense>

#include <stdexcept>
#include <string>
#include <vector>

namespace Eigen {
template <>
struct NumTraits<qd_real> : GenericNumTraits<qd_real> {
  using Real = qd_real;
  using NonInteger = qd_real;
  using Nested = qd_real;
  using Literal = qd_real;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 8,
    AddCost = 8,
    MulCost = 16,
  };

  static inline Real epsilon() { return qd_real("1e-60"); }
  static inline Real dummy_precision() { return qd_real("1e-50"); }
  static inline Real highest() { return qd_real("1e4900"); }
  static inline Real lowest() { return qd_real("-1e4900"); }
};
}  // namespace Eigen

namespace high_precision_felsenstein {

using MatrixXq = Eigen::Matrix<qd_real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXq = Eigen::Matrix<qd_real, Eigen::Dynamic, 1>;

struct LikelihoodInput {
  std::vector<int> parent;
  std::vector<qd_real> branch_length;
  std::vector<int> leaf_nodes;
  MatrixXq leaf_partials;
  MatrixXq rate_matrix;
  VectorXq root_prior;
};

class HighPrecisionFelsenstein {
 public:
  explicit HighPrecisionFelsenstein(LikelihoodInput input);

  qd_real ComputeLogLikelihood() const;

 private:
  struct NodeState {
    VectorXq partial;
  };

  LikelihoodInput input_;
  int root_ = -1;
  int num_states_ = 0;
  std::vector<std::vector<int>> children_;
  std::vector<bool> is_leaf_;
  std::vector<int> postorder_;

  void ValidateInput() const;
  void BuildTreeCache();
  void BuildPostorder();

  VectorXq ApplyTransitionUniformization(qd_real branch_length, const VectorXq& child_partial) const;
  static qd_real MaxAbs(const VectorXq& vec);
};

qd_real ParseQd(const std::string& token);

}  // namespace high_precision_felsenstein
