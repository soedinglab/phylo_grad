#include "felsenstein_qd.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace hpf = high_precision_felsenstein;

namespace {

qd_real SampleQd(std::mt19937_64* rng, const qd_real& lo, const qd_real& hi) {
  const double lo_d = to_double(lo);
  const double hi_d = to_double(hi);
  std::uniform_real_distribution<double> dist(lo_d, hi_d);
  return qd_real(dist(*rng));
}

qd_real AbsQd(const qd_real& x) {
  return (x < qd_real(0.0)) ? -x : x;
}

qd_real CentralDiffStep(const qd_real& x) {
  const qd_real base = qd_real("1e-6");
  const qd_real ax = AbsQd(x);
  return (ax > qd_real(1.0)) ? (base * ax) : base;
}

qd_real SafePositiveStep(const qd_real& x) {
  qd_real eps = CentralDiffStep(x);
  if (x - eps <= qd_real(0.0)) {
    eps = x * qd_real("0.49");
  }
  if (eps <= qd_real(0.0)) {
    throw std::runtime_error("failed to construct positive central-difference step");
  }
  return eps;
}

size_t Pow2Checked(int exp) {
  size_t out = 1;
  for (int i = 0; i < exp; ++i) {
    if (out > std::numeric_limits<size_t>::max() / 2) {
      throw std::invalid_argument("tree_height is too large");
    }
    out *= 2;
  }
  return out;
}

void BuildRandomFullBinaryTree(
    int tree_height,
    qd_real min_branch_length,
    qd_real max_branch_length,
    std::mt19937_64* rng,
    std::vector<int>* parent,
    std::vector<qd_real>* branch_length,
    std::vector<int>* leaf_nodes) {
  const size_t num_leaves = Pow2Checked(tree_height);
  const size_t num_nodes = num_leaves * 2 - 1;
  if (num_nodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("generated tree is too large for int indexing");
  }

  parent->assign(num_nodes, -1);
  branch_length->assign(num_nodes, qd_real(0.0));

  for (size_t node = 0; node < num_nodes; ++node) {
    const size_t left = node * 2 + 1;
    const size_t right = left + 1;
    if (left < num_nodes) {
      (*parent)[left] = static_cast<int>(node);
    }
    if (right < num_nodes) {
      (*parent)[right] = static_cast<int>(node);
    }
  }

  for (size_t node = 1; node < num_nodes; ++node) {
    (*branch_length)[node] = SampleQd(rng, min_branch_length, max_branch_length);
  }

  leaf_nodes->clear();
  leaf_nodes->reserve(num_leaves);
  const size_t first_leaf = num_leaves - 1;
  for (size_t node = first_leaf; node < num_nodes; ++node) {
    leaf_nodes->push_back(static_cast<int>(node));
  }
}

hpf::MatrixXq BuildRandomLeafPartials(
    int num_leaves,
    int num_states,
    qd_real min_leaf_partial,
    qd_real max_leaf_partial,
    std::mt19937_64* rng) {
  hpf::MatrixXq leaf_partials(num_leaves, num_states);
  for (int r = 0; r < num_leaves; ++r) {
    for (int c = 0; c < num_states; ++c) {
      leaf_partials(r, c) = SampleQd(rng, min_leaf_partial, max_leaf_partial);
    }
  }
  return leaf_partials;
}

hpf::MatrixXq BuildRandomSymmetricS(int num_states, std::mt19937_64* rng) {
  hpf::MatrixXq s = hpf::MatrixXq::Zero(num_states, num_states);
  for (int i = 0; i < num_states; ++i) {
    for (int j = i + 1; j < num_states; ++j) {
      const qd_real v = SampleQd(rng, qd_real("0.10"), qd_real("1.00"));
      s(i, j) = v;
      s(j, i) = v;
    }
  }
  return s;
}

hpf::VectorXq BuildRandomSqrtPi(int num_states, std::mt19937_64* rng) {
  hpf::VectorXq sqrt_pi(num_states);
  qd_real norm2 = qd_real(0.0);
  for (int i = 0; i < num_states; ++i) {
    const qd_real v = SampleQd(rng, qd_real("0.10"), qd_real("1.00"));
    sqrt_pi[i] = v;
    norm2 += v * v;
  }

  const qd_real inv_norm = qd_real(1.0) / sqrt(norm2);
  for (int i = 0; i < num_states; ++i) {
    sqrt_pi[i] *= inv_norm;
  }
  return sqrt_pi;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: high_precision_felsenstein <tree_height> <num_states> [seed]\n";
    return 1;
  }

  try {
    const int tree_height = std::stoi(argv[1]);
    const int num_states = std::stoi(argv[2]);
    const unsigned int seed = (argc == 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 1u;

    if (tree_height < 1) {
      throw std::invalid_argument("tree_height must be >= 1");
    }
    if (num_states < 2) {
      throw std::invalid_argument("num_states must be >= 2");
    }

    std::mt19937_64 rng(seed);

    std::vector<int> parent;
    std::vector<qd_real> branch_length;
    std::vector<int> leaf_nodes;

    BuildRandomFullBinaryTree(
        tree_height,
        qd_real("0.01"),
        qd_real("0.20"),
        &rng,
        &parent,
        &branch_length,
        &leaf_nodes);

    const hpf::MatrixXq leaf_partials = BuildRandomLeafPartials(
        static_cast<int>(leaf_nodes.size()),
        num_states,
        qd_real("0.10"),
        qd_real("1.00"),
        &rng);

    const hpf::MatrixXq symmetric_s = BuildRandomSymmetricS(num_states, &rng);
    const hpf::VectorXq sqrt_pi = BuildRandomSqrtPi(num_states, &rng);

    const hpf::LikelihoodInput input{
        .parent = parent,
        .branch_length = branch_length,
        .leaf_nodes = leaf_nodes,
        .leaf_partials = leaf_partials,
        .symmetric_s = symmetric_s,
        .sqrt_pi = sqrt_pi,
    };

    const hpf::GradientOutput out = hpf::ComputeColumnLogLikelihoodAndGradients(input);
    const qd_real log_likelihood = out.log_likelihood;
    const qd_real likelihood = exp(log_likelihood);

    qd_real max_abs_err_sqrt_pi = qd_real(0.0);
    qd_real max_rel_err_sqrt_pi = qd_real(0.0);
    int worst_sqrt_pi = -1;

    for (int m = 0; m < num_states; ++m) {
      hpf::LikelihoodInput plus_input = input;
      hpf::LikelihoodInput minus_input = input;

      const qd_real eps = SafePositiveStep(input.sqrt_pi[m]);
      plus_input.sqrt_pi[m] += eps;
      minus_input.sqrt_pi[m] -= eps;

      const qd_real f_plus = hpf::ComputeColumnLogLikelihood(plus_input);
      const qd_real f_minus = hpf::ComputeColumnLogLikelihood(minus_input);
      const qd_real fd = (f_plus - f_minus) / (qd_real(2.0) * eps);
      const qd_real analytic = out.grad_sqrt_pi[m];
      const qd_real abs_err = AbsQd(fd - analytic);
      const qd_real denom = AbsQd(fd) + AbsQd(analytic) + qd_real("1e-30");
      const qd_real rel_err = abs_err / denom;

      if (abs_err > max_abs_err_sqrt_pi) {
        max_abs_err_sqrt_pi = abs_err;
        worst_sqrt_pi = m;
      }
      if (rel_err > max_rel_err_sqrt_pi) {
        max_rel_err_sqrt_pi = rel_err;
      }
    }

    qd_real max_abs_err_s = qd_real(0.0);
    qd_real max_rel_err_s = qd_real(0.0);
    int worst_s_i = -1;
    int worst_s_j = -1;

    for (int i = 0; i < num_states; ++i) {
      for (int j = i + 1; j < num_states; ++j) {
        hpf::LikelihoodInput plus_input = input;
        hpf::LikelihoodInput minus_input = input;

        const qd_real eps = SafePositiveStep(input.symmetric_s(i, j));

        plus_input.symmetric_s(i, j) += eps;
        plus_input.symmetric_s(j, i) += eps;
        minus_input.symmetric_s(i, j) -= eps;
        minus_input.symmetric_s(j, i) -= eps;

        const qd_real f_plus = hpf::ComputeColumnLogLikelihood(plus_input);
        const qd_real f_minus = hpf::ComputeColumnLogLikelihood(minus_input);
        const qd_real fd = (f_plus - f_minus) / (qd_real(2.0) * eps);
        const qd_real analytic = out.grad_symmetric_s(i, j);
        const qd_real abs_err = AbsQd(fd - analytic);
        const qd_real denom = AbsQd(fd) + AbsQd(analytic) + qd_real("1e-30");
        const qd_real rel_err = abs_err / denom;

        if (abs_err > max_abs_err_s) {
          max_abs_err_s = abs_err;
          worst_s_i = i;
          worst_s_j = j;
        }
        if (rel_err > max_rel_err_s) {
          max_rel_err_s = rel_err;
        }
      }
    }

    std::cout << std::setprecision(70);
    std::cout << "log_likelihood\t" << log_likelihood << "\n";
    std::cout << "likelihood\t" << likelihood << "\n";
    std::cout << "grad_sqrt_pi\t";
    for (int i = 0; i < out.grad_sqrt_pi.size(); ++i) {
      if (i > 0) {
        std::cout << ",";
      }
      std::cout << out.grad_sqrt_pi[i];
    }
    std::cout << "\n";

    std::cout << "grad_symmetric_s_upper\t";
    bool first = true;
    for (int i = 0; i < out.grad_symmetric_s.rows(); ++i) {
      for (int j = i + 1; j < out.grad_symmetric_s.cols(); ++j) {
        if (!first) {
          std::cout << ",";
        }
        first = false;
        std::cout << "(" << i << "," << j << ")=" << out.grad_symmetric_s(i, j);
      }
    }
    std::cout << "\n";

    std::cout << "fdcheck_sqrt_pi\t"
              << "max_abs_err=" << max_abs_err_sqrt_pi
              << ",max_rel_err=" << max_rel_err_sqrt_pi
              << ",worst_index=" << worst_sqrt_pi << "\n";

    std::cout << "fdcheck_symmetric_s\t"
              << "max_abs_err=" << max_abs_err_s
              << ",max_rel_err=" << max_rel_err_s
              << ",worst_pair=(" << worst_s_i << "," << worst_s_j << ")\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
