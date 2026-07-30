#include "felsenstein_qd.hpp"

#include <functional>

#include <unsupported/Eigen/MatrixFunctions>

namespace high_precision_felsenstein
{

  qd_real ParseQd(const std::string &token)
  {
    return qd_real(token.c_str());
  }

  namespace
  {

    struct MaterializedInput
    {
      std::vector<int> parent;
      std::vector<qd_real> branch_length;
      std::vector<int> leaf_nodes;
      MatrixXq leaf_partials;
      MatrixXq rate_matrix;
      VectorXq root_prior;
    };

    struct TreeCache
    {
      int root = -1;
      int num_states = 0;
      std::vector<std::vector<int>> children;
      std::vector<bool> is_leaf;
      std::vector<int> postorder;
      std::vector<MatrixXq> transition_matrices;
    };

    struct ForwardPassResult
    {
      std::vector<VectorXq> up;
      std::vector<VectorXq> transformed;
      qd_real root_likelihood;
      qd_real log_likelihood;
    };

    using MatrixXqCol = Eigen::Matrix<qd_real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    MatrixXq EigenMatrixExp(const MatrixXq &a)
    {
      if (a.rows() != a.cols())
      {
        throw std::invalid_argument("EigenMatrixExp expects a square matrix");
      }

      // Eigen's matrix-function internals are more robust on ColMajor temporaries.
      const MatrixXqCol a_col = a;
      const MatrixXqCol exp_col = a_col.exp();
      return MatrixXq(exp_col);
    }

    void ValidateModelInput(const LikelihoodInput &input)
    {
      if (input.symmetric_s.rows() != input.symmetric_s.cols() || input.symmetric_s.rows() == 0)
      {
        throw std::invalid_argument("symmetric_s must be square and non-empty");
      }

      if (input.sqrt_pi.size() != input.symmetric_s.rows())
      {
        throw std::invalid_argument("sqrt_pi size must match symmetric_s dimension");
      }
    }

    MatrixXq BuildRateMatrixFromSAndSqrtPi(const MatrixXq &symmetric_s, const VectorXq &sqrt_pi)
    {
      const int k = static_cast<int>(sqrt_pi.size());
      MatrixXq q = MatrixXq::Zero(k, k);

      for (int i = 0; i < k; ++i)
      {
        if (sqrt_pi[i] <= qd_real(0.0))
        {
          throw std::invalid_argument("sqrt_pi entries must be strictly positive");
        }
      }

      for (int i = 0; i < k; ++i)
      {
        qd_real row_sum = qd_real(0.0);
        for (int j = 0; j < k; ++j)
        {
          if (i == j)
          {
            continue;
          }

          if (symmetric_s(i, j) != symmetric_s(j, i))
          {
            throw std::invalid_argument("symmetric_s must be symmetric");
          }

          if (symmetric_s(i, j) < qd_real(0.0))
          {
            throw std::invalid_argument("symmetric_s off-diagonal entries must be non-negative");
          }

          const qd_real qij = symmetric_s(i, j) * (sqrt_pi[j] / sqrt_pi[i]);
          q(i, j) = qij;
          row_sum += qij;
        }
        q(i, i) = -row_sum;
      }

      return q;
    }

    VectorXq BuildRootPriorFromSqrtPi(const VectorXq &sqrt_pi)
    {
      const int k = static_cast<int>(sqrt_pi.size());
      VectorXq pi(k);

      qd_real z = qd_real(0.0);
      for (int i = 0; i < k; ++i)
      {
        const qd_real v = sqrt_pi[i] * sqrt_pi[i];
        pi[i] = v;
        z += v;
      }

      if (z <= qd_real(0.0))
      {
        throw std::invalid_argument("sqrt_pi implies non-positive normalization");
      }

      for (int i = 0; i < k; ++i)
      {
        pi[i] /= z;
      }
      return pi;
    }

    MaterializedInput BuildMaterializedInput(const LikelihoodInput &input)
    {
      ValidateModelInput(input);

      MaterializedInput out;
      out.parent = input.parent;
      out.branch_length = input.branch_length;
      out.leaf_nodes = input.leaf_nodes;
      out.leaf_partials = input.leaf_partials;

      out.rate_matrix = BuildRateMatrixFromSAndSqrtPi(input.symmetric_s, input.sqrt_pi);
      out.root_prior = BuildRootPriorFromSqrtPi(input.sqrt_pi);
      return out;
    }

    void ValidateInput(const MaterializedInput &input)
    {
      const int n = static_cast<int>(input.parent.size());
      if (n == 0)
      {
        throw std::invalid_argument("parent list must not be empty");
      }

      if (static_cast<int>(input.branch_length.size()) != n)
      {
        throw std::invalid_argument("branch_length must have same length as parent");
      }

      for (int node = 0; node < n; ++node)
      {
        if (input.branch_length[node] < qd_real(0.0))
        {
          throw std::invalid_argument("branch_length entries must be non-negative");
        }
      }

      if (input.rate_matrix.rows() != input.rate_matrix.cols() || input.rate_matrix.rows() == 0)
      {
        throw std::invalid_argument("rate_matrix must be square and non-empty");
      }

      if (input.leaf_partials.cols() != input.rate_matrix.rows())
      {
        throw std::invalid_argument("leaf_partials column count must match number of states");
      }

      if (input.root_prior.size() != input.rate_matrix.rows())
      {
        throw std::invalid_argument("root_prior size must match number of states");
      }

      if (static_cast<int>(input.leaf_nodes.size()) != input.leaf_partials.rows())
      {
        throw std::invalid_argument("leaf_nodes count must match rows in leaf_partials");
      }

      int root_count = 0;
      for (int child = 0; child < n; ++child)
      {
        const int p = input.parent[child];
        if (p == -1)
        {
          ++root_count;
          continue;
        }
        if (p < 0 || p >= n)
        {
          throw std::invalid_argument("parent index out of range");
        }
      }

      if (root_count != 1)
      {
        throw std::invalid_argument("tree must contain exactly one root with parent -1");
      }
    }

    TreeCache BuildTreeCache(const MaterializedInput &input)
    {
      const int n = static_cast<int>(input.parent.size());
      TreeCache cache;
      cache.num_states = static_cast<int>(input.rate_matrix.rows());

      cache.children.assign(n, {});
      cache.is_leaf.assign(n, true);

      for (int child = 0; child < n; ++child)
      {
        const int p = input.parent[child];
        if (p == -1)
        {
          cache.root = child;
        }
        else
        {
          cache.children[p].push_back(child);
          cache.is_leaf[p] = false;
        }
      }

      for (int leaf_node : input.leaf_nodes)
      {
        if (leaf_node < 0 || leaf_node >= n)
        {
          throw std::invalid_argument("leaf node index out of range");
        }
        if (!cache.is_leaf[leaf_node])
        {
          throw std::invalid_argument("leaf partials contain an internal node index");
        }
      }

      return cache;
    }

    void BuildPostorder(const MaterializedInput &input, TreeCache *cache)
    {
      cache->postorder.clear();
      cache->postorder.reserve(input.parent.size());

      std::vector<bool> visited(input.parent.size(), false);
      std::vector<bool> in_stack(input.parent.size(), false);

      std::function<void(int)> dfs = [&](int node)
      {
        if (in_stack[node])
        {
          throw std::invalid_argument("cycle detected in parent list");
        }
        if (visited[node])
        {
          return;
        }

        in_stack[node] = true;
        for (int child : cache->children[node])
        {
          dfs(child);
        }
        in_stack[node] = false;

        visited[node] = true;
        cache->postorder.push_back(node);
      };

      dfs(cache->root);

      if (cache->postorder.size() != input.parent.size())
      {
        throw std::invalid_argument("tree is disconnected from root");
      }
    }

    void BuildTransitionMatrices(const MaterializedInput &input, TreeCache *cache)
    {
      const int n = static_cast<int>(input.parent.size());
      cache->transition_matrices.assign(n, MatrixXq::Identity(cache->num_states, cache->num_states));

      for (int node = 0; node < n; ++node)
      {
        if (node == cache->root)
        {
          continue;
        }
        const qd_real t = input.branch_length[node];
        if (t == qd_real(0.0))
        {
          continue;
        }
        cache->transition_matrices[node] = EigenMatrixExp(input.rate_matrix * t);
      }
    }

    void ValidateLeafPartials(const MaterializedInput &input, const TreeCache &cache, const MatrixXq &leaf_partials)
    {
      if (leaf_partials.rows() != static_cast<int>(input.leaf_nodes.size()))
      {
        throw std::invalid_argument("leaf_partials row count must match leaf_nodes count");
      }
      if (leaf_partials.cols() != cache.num_states)
      {
        throw std::invalid_argument("leaf_partials column count must match number of states");
      }
    }

    qd_real ComputeLogLikelihoodForLeafPartials(
        const MaterializedInput &input,
        const TreeCache &cache,
        const MatrixXq &leaf_partials)
    {
      ValidateLeafPartials(input, cache, leaf_partials);

      const int n = static_cast<int>(input.parent.size());

      struct NodeState
      {
        VectorXq partial;
      };

      std::vector<NodeState> states(static_cast<size_t>(n));
      for (int node = 0; node < n; ++node)
      {
        states[node].partial = VectorXq::Ones(cache.num_states);
      }

      for (int leaf_row = 0; leaf_row < static_cast<int>(input.leaf_nodes.size()); ++leaf_row)
      {
        const int node = input.leaf_nodes[leaf_row];
        states[node].partial = leaf_partials.row(leaf_row).transpose();
      }

      for (int node : cache.postorder)
      {
        if (node == cache.root)
        {
          continue;
        }
        const int parent = input.parent[node];

        const VectorXq transformed = cache.transition_matrices[node] * states[node].partial;
        states[parent].partial.array() *= transformed.array();
      }

      const qd_real root_likelihood = input.root_prior.dot(states[cache.root].partial);
      if (root_likelihood <= qd_real(0.0))
      {
        throw std::runtime_error("root likelihood is non-positive");
      }

      return log(root_likelihood);
    }

    ForwardPassResult ComputeForwardPass(
        const MaterializedInput &input,
        const TreeCache &cache,
        const MatrixXq &leaf_partials)
    {
      ValidateLeafPartials(input, cache, leaf_partials);

      const int n = static_cast<int>(input.parent.size());
      ForwardPassResult out;
      out.up.resize(static_cast<size_t>(n));
      out.transformed.resize(static_cast<size_t>(n));
      for (int node = 0; node < n; ++node)
      {
        out.up[node] = VectorXq::Ones(cache.num_states);
        out.transformed[node] = VectorXq::Ones(cache.num_states);
      }

      for (int leaf_row = 0; leaf_row < static_cast<int>(input.leaf_nodes.size()); ++leaf_row)
      {
        const int node = input.leaf_nodes[leaf_row];
        out.up[node] = leaf_partials.row(leaf_row).transpose();
      }

      for (int node : cache.postorder)
      {
        if (node == cache.root)
        {
          continue;
        }
        const int parent = input.parent[node];
        out.transformed[node] = cache.transition_matrices[node] * out.up[node];
        out.up[parent].array() *= out.transformed[node].array();
      }

      out.root_likelihood = input.root_prior.dot(out.up[cache.root]);
      if (out.root_likelihood <= qd_real(0.0))
      {
        throw std::runtime_error("root likelihood is non-positive");
      }
      out.log_likelihood = log(out.root_likelihood);
      return out;
    }

    std::vector<MatrixXq> ComputeBranchTransitionAdjoints(
        const MaterializedInput &input,
        const TreeCache &cache,
        const ForwardPassResult &forward)
    {
      const int n = static_cast<int>(input.parent.size());
      const int k = cache.num_states;

      std::vector<VectorXq> adj_up(static_cast<size_t>(n), VectorXq::Zero(k));
      std::vector<MatrixXq> grad_transition(static_cast<size_t>(n), MatrixXq::Zero(k, k));

      adj_up[cache.root] = input.root_prior;

      for (int idx = static_cast<int>(cache.postorder.size()) - 1; idx >= 0; --idx)
      {
        const int parent = cache.postorder[idx];
        const std::vector<int> &children = cache.children[parent];
        if (children.empty())
        {
          continue;
        }

        for (int child : children)
        {
          VectorXq sibling_product = VectorXq::Ones(k);
          for (int other_child : children)
          {
            if (other_child == child)
            {
              continue;
            }
            sibling_product.array() *= forward.transformed[other_child].array();
          }

          const VectorXq dL_dt = adj_up[parent].array() * sibling_product.array();
          grad_transition[child] = dL_dt * forward.up[child].transpose();
          adj_up[child] += cache.transition_matrices[child].transpose() * dL_dt;
        }
      }

      return grad_transition;
    }

    MatrixXq FrechetExpViaBlockMatrix(const MatrixXq &a, const MatrixXq &e)
    {
      if (a.rows() != a.cols())
      {
        throw std::invalid_argument("FrechetExpViaBlockMatrix expects square A");
      }
      if (e.rows() != a.rows() || e.cols() != a.cols())
      {
        throw std::invalid_argument("FrechetExpViaBlockMatrix expects E with same shape as A");
      }

      const int k = a.rows();
      MatrixXqCol block = MatrixXqCol::Zero(2 * k, 2 * k);
      block.topLeftCorner(k, k) = a;
      block.topRightCorner(k, k) = e;
      block.bottomRightCorner(k, k) = a;

      const MatrixXqCol exp_block = block.exp();
      return MatrixXq(exp_block.topRightCorner(k, k));
    }

    MatrixXq BuildDqDsParam(const MatrixXq &symmetric_s, const VectorXq &sqrt_pi, int i, int j)
    {
      const int k = static_cast<int>(sqrt_pi.size());
      MatrixXq d_q = MatrixXq::Zero(k, k);

      const qd_real r_ij = sqrt_pi[j] / sqrt_pi[i];
      const qd_real r_ji = sqrt_pi[i] / sqrt_pi[j];

      d_q(i, j) = r_ij;
      d_q(i, i) -= r_ij;
      d_q(j, i) = r_ji;
      d_q(j, j) -= r_ji;
      (void)symmetric_s;
      return d_q;
    }

    MatrixXq BuildDqDSqrtPiParam(const MatrixXq &symmetric_s, const VectorXq &sqrt_pi, int m)
    {
      const int k = static_cast<int>(sqrt_pi.size());
      MatrixXq d_q = MatrixXq::Zero(k, k);

      for (int a = 0; a < k; ++a)
      {
        for (int b = 0; b < k; ++b)
        {
          if (a == b)
          {
            continue;
          }

          qd_real dq_ab = qd_real(0.0);
          if (b == m)
          {
            dq_ab += symmetric_s(a, b) / sqrt_pi[a];
          }
          if (a == m)
          {
            dq_ab -= symmetric_s(a, b) * sqrt_pi[b] / (sqrt_pi[a] * sqrt_pi[a]);
          }
          d_q(a, b) = dq_ab;
        }
      }

      for (int a = 0; a < k; ++a)
      {
        qd_real row_sum = qd_real(0.0);
        for (int b = 0; b < k; ++b)
        {
          if (a == b)
          {
            continue;
          }
          row_sum += d_q(a, b);
        }
        d_q(a, a) = -row_sum;
      }
      return d_q;
    }

    VectorXq BuildRootPriorDerivativeWrtSqrtPi(const VectorXq &sqrt_pi, int m)
    {
      const int k = static_cast<int>(sqrt_pi.size());
      VectorXq d_pi = VectorXq::Zero(k);
      qd_real z = qd_real(0.0);
      for (int i = 0; i < k; ++i)
      {
        z += sqrt_pi[i] * sqrt_pi[i];
      }

      const qd_real dz = qd_real(2.0) * sqrt_pi[m];
      const qd_real z2 = z * z;
      for (int r = 0; r < k; ++r)
      {
        const qd_real nr = sqrt_pi[r] * sqrt_pi[r];
        const qd_real dnr = (r == m) ? qd_real(2.0) * sqrt_pi[r] : qd_real(0.0);
        d_pi[r] = (dnr * z - nr * dz) / z2;
      }
      return d_pi;
    }

  } // namespace

  qd_real ComputeColumnLogLikelihood(const LikelihoodInput &input)
  {
    MaterializedInput materialized = BuildMaterializedInput(input);
    ValidateInput(materialized);
    TreeCache cache = BuildTreeCache(materialized);
    BuildPostorder(materialized, &cache);
    BuildTransitionMatrices(materialized, &cache);
    return ComputeLogLikelihoodForLeafPartials(materialized, cache, materialized.leaf_partials);
  }

  GradientOutput ComputeColumnLogLikelihoodAndGradients(const LikelihoodInput &input)
  {
    MaterializedInput materialized = BuildMaterializedInput(input);
    ValidateInput(materialized);
    TreeCache cache = BuildTreeCache(materialized);
    BuildPostorder(materialized, &cache);
    BuildTransitionMatrices(materialized, &cache);

    const ForwardPassResult forward = ComputeForwardPass(materialized, cache, materialized.leaf_partials);
    const std::vector<MatrixXq> grad_transition = ComputeBranchTransitionAdjoints(materialized, cache, forward);

    const int k = static_cast<int>(input.sqrt_pi.size());

    // Pure reverse-mode: first accumulate dL/dQ from all branches, then map to model parameters.
    MatrixXq grad_q = MatrixXq::Zero(k, k);
    for (int node = 0; node < static_cast<int>(materialized.parent.size()); ++node)
    {
      if (node == cache.root)
      {
        continue;
      }
      const qd_real t = materialized.branch_length[node];
      if (t == qd_real(0.0))
      {
        continue;
      }

      const MatrixXq a = materialized.rate_matrix * t;

      // Adjoint identity:
      // <G_P, L_exp(A, E)> = <L_exp(A^T, G_P^T)^T, E>
      // so dL/dA = L_exp(A^T, G_P^T)^T.
      const MatrixXq g_a =
          FrechetExpViaBlockMatrix(a.transpose(), grad_transition[node].transpose()).transpose();

      grad_q.array() += (g_a * t).array();
    }

    MatrixXq grad_s = MatrixXq::Zero(k, k);
    VectorXq grad_sqrt_pi = VectorXq::Zero(k);

    for (int i = 0; i < k; ++i)
    {
      for (int j = i + 1; j < k; ++j)
      {
        const MatrixXq d_q = BuildDqDsParam(input.symmetric_s, input.sqrt_pi, i, j);
        const qd_real dL = (grad_q.array() * d_q.array()).sum();
        const qd_real dlog = dL / forward.root_likelihood;
        grad_s(i, j) = dlog;
        grad_s(j, i) = dlog;
      }
    }

    for (int m = 0; m < k; ++m)
    {
      qd_real dL = qd_real(0.0);

      const VectorXq d_pi = BuildRootPriorDerivativeWrtSqrtPi(input.sqrt_pi, m);
      dL += d_pi.dot(forward.up[cache.root]);

      const MatrixXq d_q = BuildDqDSqrtPiParam(input.symmetric_s, input.sqrt_pi, m);
      dL += (grad_q.array() * d_q.array()).sum();

      grad_sqrt_pi[m] = dL / forward.root_likelihood;
    }

    return GradientOutput{
        .log_likelihood = forward.log_likelihood,
        .grad_symmetric_s = std::move(grad_s),
        .grad_sqrt_pi = std::move(grad_sqrt_pi),
    };
  }
} // namespace high_precision_felsenstein
