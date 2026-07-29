#include "felsenstein_qd.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

namespace high_precision_felsenstein {

namespace {

constexpr int kMaxUniformizationTerms = 2048;

}  // namespace

qd_real ParseQd(const std::string& token) {
  return qd_real(token.c_str());
}

HighPrecisionFelsenstein::HighPrecisionFelsenstein(LikelihoodInput input) : input_(std::move(input)) {
  ValidateInput();
  BuildTreeCache();
  BuildPostorder();
}

void HighPrecisionFelsenstein::ValidateInput() const {
  const int n = static_cast<int>(input_.parent.size());
  if (n == 0) {
    throw std::invalid_argument("parent list must not be empty");
  }

  if (static_cast<int>(input_.branch_length.size()) != n) {
    throw std::invalid_argument("branch_length must have same length as parent");
  }

  if (input_.rate_matrix.rows() != input_.rate_matrix.cols() || input_.rate_matrix.rows() == 0) {
    throw std::invalid_argument("rate_matrix must be square and non-empty");
  }

  if (input_.leaf_partials.cols() != input_.rate_matrix.rows()) {
    throw std::invalid_argument("leaf_partials column count must match number of states");
  }

  if (input_.root_prior.size() != input_.rate_matrix.rows()) {
    throw std::invalid_argument("root_prior size must match number of states");
  }

  if (static_cast<int>(input_.leaf_nodes.size()) != input_.leaf_partials.rows()) {
    throw std::invalid_argument("leaf_nodes count must match rows in leaf_partials");
  }

  int root_count = 0;
  for (int child = 0; child < n; ++child) {
    const int p = input_.parent[child];
    if (p == -1) {
      ++root_count;
      continue;
    }
    if (p < 0 || p >= n) {
      throw std::invalid_argument("parent index out of range");
    }
  }

  if (root_count != 1) {
    throw std::invalid_argument("tree must contain exactly one root with parent -1");
  }
}

void HighPrecisionFelsenstein::BuildTreeCache() {
  const int n = static_cast<int>(input_.parent.size());
  num_states_ = static_cast<int>(input_.rate_matrix.rows());

  children_.assign(n, {});
  is_leaf_.assign(n, true);

  for (int child = 0; child < n; ++child) {
    const int p = input_.parent[child];
    if (p == -1) {
      root_ = child;
    } else {
      children_[p].push_back(child);
      is_leaf_[p] = false;
    }
  }

  for (int leaf_node : input_.leaf_nodes) {
    if (leaf_node < 0 || leaf_node >= n) {
      throw std::invalid_argument("leaf node index out of range");
    }
    if (!is_leaf_[leaf_node]) {
      throw std::invalid_argument("leaf partials contain an internal node index");
    }
  }
}

void HighPrecisionFelsenstein::BuildPostorder() {
  postorder_.clear();
  postorder_.reserve(input_.parent.size());

  std::vector<bool> visited(input_.parent.size(), false);
  std::vector<bool> in_stack(input_.parent.size(), false);

  std::function<void(int)> dfs = [&](int node) {
    if (in_stack[node]) {
      throw std::invalid_argument("cycle detected in parent list");
    }
    if (visited[node]) {
      return;
    }

    in_stack[node] = true;
    for (int child : children_[node]) {
      dfs(child);
    }
    in_stack[node] = false;

    visited[node] = true;
    postorder_.push_back(node);
  };

  dfs(root_);

  if (postorder_.size() != input_.parent.size()) {
    throw std::invalid_argument("tree is disconnected from root");
  }
}

qd_real HighPrecisionFelsenstein::MaxAbs(const VectorXq& vec) {
  qd_real max_val = qd_real(0.0);
  for (int i = 0; i < vec.size(); ++i) {
    const qd_real cur = abs(vec[i]);
    if (cur > max_val) {
      max_val = cur;
    }
  }
  return max_val;
}

VectorXq HighPrecisionFelsenstein::ApplyTransitionUniformization(
    qd_real branch_length, const VectorXq& child_partial) const {
  if (branch_length <= qd_real(0.0)) {
    return child_partial;
  }

  qd_real mu = qd_real(0.0);
  for (int i = 0; i < input_.rate_matrix.rows(); ++i) {
    const qd_real candidate = -input_.rate_matrix(i, i);
    if (candidate > mu) {
      mu = candidate;
    }
  }

  if (mu <= qd_real(0.0)) {
    return child_partial;
  }

  const MatrixXq identity = MatrixXq::Identity(num_states_, num_states_);
  const MatrixXq R = identity + (input_.rate_matrix / mu);
  const qd_real lambda = mu * branch_length;

  qd_real weight = exp(-lambda);
  VectorXq term = child_partial;
  VectorXq accum = weight * term;

  const qd_real tol = qd_real("1e-80");
  for (int k = 1; k <= kMaxUniformizationTerms; ++k) {
    term = R * term;
    weight *= lambda / qd_real(static_cast<double>(k));
    accum += weight * term;

    const qd_real term_bound = abs(weight) * MaxAbs(term);
    const qd_real accum_bound = MaxAbs(accum);
    if (accum_bound > qd_real(0.0) && term_bound <= tol * accum_bound) {
      break;
    }

    if (k == kMaxUniformizationTerms) {
      throw std::runtime_error("uniformization did not converge within max terms");
    }
  }

  return accum;
}

qd_real HighPrecisionFelsenstein::ComputeLogLikelihood() const {
  const int n = static_cast<int>(input_.parent.size());

  std::vector<NodeState> states(static_cast<size_t>(n));
  for (int node = 0; node < n; ++node) {
    states[node].partial = VectorXq::Ones(num_states_);
  }

  for (int leaf_row = 0; leaf_row < static_cast<int>(input_.leaf_nodes.size()); ++leaf_row) {
    const int node = input_.leaf_nodes[leaf_row];
    states[node].partial = input_.leaf_partials.row(leaf_row).transpose();
  }

  for (int node : postorder_) {
    if (node == root_) {
      continue;
    }
    const int parent = input_.parent[node];

    const VectorXq transformed = ApplyTransitionUniformization(input_.branch_length[node], states[node].partial);
    states[parent].partial.array() *= transformed.array();
  }

  const qd_real root_likelihood = input_.root_prior.dot(states[root_].partial);
  if (root_likelihood <= qd_real(0.0)) {
    throw std::runtime_error("root likelihood is non-positive");
  }

  return log(root_likelihood);
}

}  // namespace high_precision_felsenstein
