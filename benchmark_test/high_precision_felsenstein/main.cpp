#include "felsenstein_qd.hpp"

#include <cctype>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hpf = high_precision_felsenstein;

namespace {

struct ParsedTree {
  std::vector<int> parent;
  std::vector<qd_real> branch_length;
  std::vector<std::string> node_name;
  int root = -1;
};

struct MsaData {
  std::vector<std::string> taxon_names;
  std::vector<std::string> sequences;
  int num_sites = 0;
};

void SkipWs(const std::string& s, size_t* i) {
  while (*i < s.size() && std::isspace(static_cast<unsigned char>(s[*i]))) {
    ++(*i);
  }
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool IsLabelChar(char c) {
  return c != '(' && c != ')' && c != ',' && c != ':' && c != ';' && !std::isspace(static_cast<unsigned char>(c));
}

std::string ParseLabel(const std::string& s, size_t* i) {
  SkipWs(s, i);
  size_t start = *i;
  while (*i < s.size() && IsLabelChar(s[*i])) {
    ++(*i);
  }
  if (*i == start) {
    return "";
  }
  return s.substr(start, *i - start);
}

qd_real ParseBranchLength(const std::string& s, size_t* i) {
  SkipWs(s, i);
  if (*i >= s.size() || s[*i] != ':') {
    return qd_real(0.0);
  }
  ++(*i);
  SkipWs(s, i);
  size_t start = *i;
  while (*i < s.size()) {
    const char c = s[*i];
    if (c == ',' || c == ')' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
      break;
    }
    ++(*i);
  }
  if (*i == start) {
    throw std::runtime_error("missing branch length value after ':'");
  }
  return hpf::ParseQd(s.substr(start, *i - start));
}

int AddNode(ParsedTree* tree) {
  tree->parent.push_back(-2);
  tree->branch_length.push_back(qd_real(0.0));
  tree->node_name.emplace_back();
  return static_cast<int>(tree->parent.size()) - 1;
}

int ParseSubtree(const std::string& s, size_t* i, ParsedTree* tree) {
  SkipWs(s, i);
  if (*i >= s.size()) {
    throw std::runtime_error("unexpected end while parsing Newick subtree");
  }

  if (s[*i] == '(') {
    ++(*i);
    const int node = AddNode(tree);
    while (true) {
      SkipWs(s, i);
      if (*i >= s.size()) {
        throw std::runtime_error("unexpected end while parsing internal node children");
      }

      const int child = ParseSubtree(s, i, tree);
      tree->parent[child] = node;

      SkipWs(s, i);
      if (*i >= s.size()) {
        throw std::runtime_error("unexpected end after child subtree");
      }

      if (s[*i] == ',') {
        ++(*i);
      } else if (s[*i] == ')') {
        ++(*i);
        break;
      } else {
        throw std::runtime_error("expected ',' or ')' while parsing internal node");
      }
    }

    tree->node_name[node] = ParseLabel(s, i);
    tree->branch_length[node] = ParseBranchLength(s, i);
    return node;
  }

  const int leaf = AddNode(tree);
  tree->node_name[leaf] = ParseLabel(s, i);
  if (tree->node_name[leaf].empty()) {
    throw std::runtime_error("leaf without a label in Newick tree");
  }
  tree->branch_length[leaf] = ParseBranchLength(s, i);
  return leaf;
}

ParsedTree LoadNewickTree(const std::string& path) {
  const std::string content = ReadWholeFile(path);
  size_t i = 0;
  ParsedTree tree;
  const int root = ParseSubtree(content, &i, &tree);
  tree.root = root;
  tree.parent[root] = -1;

  SkipWs(content, &i);
  if (i >= content.size() || content[i] != ';') {
    throw std::runtime_error("Newick tree must end with ';'");
  }
  ++i;
  SkipWs(content, &i);
  if (i != content.size()) {
    throw std::runtime_error("unexpected trailing content after Newick ';'");
  }

  return tree;
}

MsaData LoadFastaMsa(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }

  std::vector<std::string> names;
  std::vector<std::string> seqs;
  std::unordered_map<std::string, int> index_by_name;

  std::string line;
  std::string current_name;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    if (line[0] == '>') {
      current_name = line.substr(1);
      if (current_name.empty()) {
        throw std::runtime_error("empty FASTA header encountered in " + path);
      }
      auto it = index_by_name.find(current_name);
      if (it != index_by_name.end()) {
        throw std::runtime_error("duplicate FASTA taxon name: " + current_name);
      }
      index_by_name[current_name] = static_cast<int>(names.size());
      names.push_back(current_name);
      seqs.emplace_back();
      continue;
    }

    if (current_name.empty()) {
      throw std::runtime_error("sequence data encountered before first FASTA header");
    }

    std::string compact;
    compact.reserve(line.size());
    for (char c : line) {
      if (!std::isspace(static_cast<unsigned char>(c))) {
        compact.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
      }
    }
    seqs.back().append(compact);
  }

  if (names.empty()) {
    throw std::runtime_error("no sequences found in " + path);
  }

  const int num_sites = static_cast<int>(seqs.front().size());
  if (num_sites == 0) {
    throw std::runtime_error("empty sequences in " + path);
  }
  for (size_t i = 0; i < seqs.size(); ++i) {
    if (static_cast<int>(seqs[i].size()) != num_sites) {
      throw std::runtime_error("FASTA sequences have unequal lengths");
    }
  }

  return MsaData{.taxon_names = std::move(names), .sequences = std::move(seqs), .num_sites = num_sites};
}

std::vector<char> AminoStates() {
  return {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};
}

std::unordered_map<char, int> BuildStateIndex(const std::vector<char>& states) {
  std::unordered_map<char, int> index;
  for (int i = 0; i < static_cast<int>(states.size()); ++i) {
    index[states[i]] = i;
  }
  return index;
}

bool IsAmbiguous(char c) {
  return c == 'N' || c == 'X' || c == '?' || c == '-' || c == 'B' || c == 'Z' || c == 'J' || c == 'U' || c == 'O';
}

hpf::MatrixXq BuildEqualRatesQ(int k) {
  hpf::MatrixXq q = hpf::MatrixXq::Zero(k, k);
  const qd_real off = qd_real(1.0) / qd_real(static_cast<double>(k - 1));
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      if (i == j) {
        continue;
      }
      q(i, j) = off;
    }
    q(i, i) = qd_real(-1.0);
  }
  return q;
}

hpf::VectorXq BuildUniformRootPrior(int k) {
  hpf::VectorXq pi(k);
  const qd_real p = qd_real(1.0) / qd_real(static_cast<double>(k));
  for (int i = 0; i < k; ++i) {
    pi[i] = p;
  }
  return pi;
}

std::vector<int> CollectLeafNodes(const ParsedTree& tree) {
  const int n = static_cast<int>(tree.parent.size());
  std::vector<int> degree(n, 0);
  for (int node = 0; node < n; ++node) {
    const int p = tree.parent[node];
    if (p >= 0) {
      degree[p] += 1;
    }
  }

  std::vector<int> leaves;
  for (int node = 0; node < n; ++node) {
    if (degree[node] == 0) {
      leaves.push_back(node);
    }
  }
  return leaves;
}

std::unordered_map<std::string, std::string> BuildSequenceByName(const MsaData& msa) {
  std::unordered_map<std::string, std::string> out;
  for (int i = 0; i < static_cast<int>(msa.taxon_names.size()); ++i) {
    out[msa.taxon_names[i]] = msa.sequences[i];
  }
  return out;
}

hpf::MatrixXq BuildLeafPartialsForSite(
    int site,
    const std::vector<int>& leaf_nodes,
    const ParsedTree& tree,
    const std::unordered_map<std::string, std::string>& seq_by_name,
    const std::unordered_map<char, int>& state_index,
    int num_states) {
  hpf::MatrixXq leaf_partials(static_cast<int>(leaf_nodes.size()), num_states);

  for (int i = 0; i < static_cast<int>(leaf_nodes.size()); ++i) {
    const int node = leaf_nodes[i];
    const std::string& label = tree.node_name[node];
    auto it = seq_by_name.find(label);
    if (it == seq_by_name.end()) {
      throw std::runtime_error("leaf '" + label + "' missing in MSA");
    }

    leaf_partials.row(i).setZero();
    const char c = it->second[site];
    auto state_it = state_index.find(c);
    if (state_it != state_index.end()) {
      leaf_partials(i, state_it->second) = qd_real(1.0);
    } else if (IsAmbiguous(c)) {
      leaf_partials.row(i).setOnes();
    } else {
      throw std::runtime_error(
          "unsupported symbol '" + std::string(1, c) + "' at taxon '" + label + "', site " + std::to_string(site));
    }
  }

  return leaf_partials;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: high_precision_felsenstein <tree.newick> <alignment.fasta>\n";
    return 1;
  }

  try {
    const std::string tree_path = argv[1];
    const std::string msa_path = argv[2];

    const ParsedTree tree = LoadNewickTree(tree_path);
    const MsaData msa = LoadFastaMsa(msa_path);
    const std::vector<char> states = AminoStates();
    const int k = static_cast<int>(states.size());
    const auto state_index = BuildStateIndex(states);
    const auto seq_by_name = BuildSequenceByName(msa);
    const std::vector<int> leaf_nodes = CollectLeafNodes(tree);

    for (int leaf : leaf_nodes) {
      const std::string& label = tree.node_name[leaf];
      if (label.empty()) {
        throw std::runtime_error("leaf without label encountered");
      }
      if (seq_by_name.find(label) == seq_by_name.end()) {
        throw std::runtime_error("leaf '" + label + "' not found in alignment");
      }
    }

    const hpf::MatrixXq rate_matrix = BuildEqualRatesQ(k);
    const hpf::VectorXq root_prior = BuildUniformRootPrior(k);

    qd_real total_log_likelihood = qd_real(0.0);
    for (int site = 0; site < msa.num_sites; ++site) {
      hpf::LikelihoodInput input{
          .parent = tree.parent,
          .branch_length = tree.branch_length,
          .leaf_nodes = leaf_nodes,
          .leaf_partials = BuildLeafPartialsForSite(site, leaf_nodes, tree, seq_by_name, state_index, k),
          .rate_matrix = rate_matrix,
          .root_prior = root_prior,
      };
      hpf::HighPrecisionFelsenstein felsenstein(std::move(input));
      total_log_likelihood += felsenstein.ComputeLogLikelihood();
    }

    const qd_real likelihood = exp(total_log_likelihood);
    std::cout << std::setprecision(70);
    std::cout << "log_likelihood\t" << total_log_likelihood << "\n";
    std::cout << "likelihood\t" << likelihood << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
