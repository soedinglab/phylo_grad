# Felsenstein_rs: efficient and parallel inference of substitution rates for phylogenetic trees
 This repository compiles into a python module `felsenstein_rs`.
 Given a phylogenetic tree and sequence alignment information, the module enables the estimation of nucleotide substitution rates for each column, as well as the substitution rates for pairs of nucleotides in a given pair of aligned columns.

## Installation
 To install the package into your current environment, run
 ```
 maturin develop --release
 ```
 Alternatively, you can build the python wheel for any version of python currently installed in your system:
 ```
 maturin build --release
 ```

### Dependencies
 By default, Cargo will take care of dependencies, in particular it will download and build the required distribution of lapack from source. To control the choice of lapack implementation, please follow [this guide](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki).

## Data format
The phylogenetic tree is read from the input file; it is assumed that the root node has at most three children, and all other nodes have at most two. The first line of the input file is skipped for technical reasons, and the rest is parsed as a CSV file with no header, where each record describes a single tree node and has the following format:

| parent_id | distance | sequence |
|:---------:|:--------:|:--------:|
| int       | double   | string   |

The tree root has parent id set to $-1$.
For non-terminal nodes, the sequence field is empty, while for the leaf nodes it contains a string representing a nucleotide sequence in [IUPAC notation](https://www.bioinformatics.org/sms/iupac.html). These strings may contain the characters A, C, G ,T, U, as well as N or - or . for gaps, and the characters R, Y, S, W, K, M, B, D, H, V for incompletely specified bases.

## Parametrization
The matrix of transition rates has the property that the sum of entries in each row is $0$. We are also assuming the transition rate matrix $R$ is time-symmetric. In this case, it may be parametrized [[1, III.B]](#references) as $R = D^{-1} S D$, where $D = diag(\sqrt{\pi})$, $\pi$ is the stationary distribution of $R$, and $S$ is a symmetric matrix. Let $S = \varDelta + D_S + \varDelta^{T}$, where $\varDelta$ is strictly upper-triangular, and $D_S$ is a diagonal matrix. We use $\varDelta$ and $\sqrt{\pi}$ as parameters for our model; if these parameters are given, notice that there is a unique choice of $D_S$ such that the sum of entries of $R$ in each row is zero.

## Interface
 The module provides functions for computing the likelihood of a phylogenetic tree given the parameters described above, as well as the gradient of the likelihood wrt the model parameters.

```
class FTree
  __init__(data_path: str, distance_threshold: double)
  infer_param(
        index_pairs: array((COLS, 2, ), dtype = np.uint),
        deltas: array((COLS, DIM, DIM, ), dtype = np.float64),
        sqrt_pi: array((COLS, DIM, ), dtype = np.float64),
        gaps: bool = false,
        p_none: array((SEQ_LENGTH, DIM, ), dtype = np.float64) = None)
  -> {
      "log_likelihood": array((COLS,), dtype=np.float64),
      "grad_delta": array((COLS, DIM, DIM, ), dtype=np.float64),
      "grad_sqrt_pi": array((COLS, DIM, ), dtype=np.float64),
      "grad_rate": array((COLS, DIM, DIM, ), dtype=np.float64)
     }

class FTreeAminoacid
  __init__(data_path: str, distance_threshold: double)
  infer_param_unpaired(
        idx: array((COLS, ), dtype = np.uint),
        deltas: array((COLS, DIM, DIM, ), dtype = np.float64),
        sqrt_pi: array((COLS, DIM, ), dtype = np.float64),
        gaps: bool = false,
        p_none: array((SEQ_LENGTH, DIM, ), dtype = np.float64) = None)
  -> {
      "log_likelihood": array((COLS,), dtype=np.float64),
      "grad_delta": array((COLS, DIM, DIM, ), dtype=np.float64),
      "grad_sqrt_pi": array((COLS, DIM, ), dtype=np.float64),
      "grad_rate": array((COLS, DIM, DIM, ), dtype=np.float64)
     }
```

## References
[1] [McGibbon R.T., Pande V.S. - Efficient maximum likelihood parameterization of continuous-time Markov processes.](https://arxiv.org/abs/1504.01804)