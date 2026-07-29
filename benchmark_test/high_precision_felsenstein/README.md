# High Precision Felsenstein (C++ / Eigen / libqd)

This folder contains a high-precision implementation of the Felsenstein pruning
algorithm using:

- `Eigen` for matrix/vector operations.
- `libqd` (`qd_real`) for quad-double precision arithmetic.

It computes likelihood only (no gradients).

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## CLI Usage

```bash
./build/high_precision_felsenstein <tree.newick> <alignment.fasta> [auto|dna|aa]
```

Examples:

```bash
./build/high_precision_felsenstein ../test_tree.newick ../data/sim/alignment_16_10.fasta
./build/high_precision_felsenstein ../test_tree.newick ../data/sim/alignment_16_10.fasta aa
```

Output fields:

- `log_likelihood`: total alignment log-likelihood.
- `likelihood`: total alignment likelihood (`exp(log_likelihood)`).

## Input Assumptions

- Newick tree must include branch lengths on edges.
- Leaf names in the Newick tree must match FASTA headers exactly.
- All FASTA sequences must have the same length.
- No rescaling is applied during pruning (as requested).

## Model Used

The binary currently builds a simple equal-rates CTMC for the chosen alphabet:

- `dna`: 4 states (`A,C,G,T`)
- `aa`: 20 amino-acid states (`ACDEFGHIKLMNPQRSTVWY`)
- `auto`: chooses `dna` if all symbols are DNA-like (`A,C,G,T,N,?,-`), otherwise `aa`

Root prior is uniform over states.

Ambiguous symbols (`N`, `X`, `?`, `-`, `B`, `Z`, `J`, `U`, `O`) are treated as
all-states-possible partials.

## Numerical Method

Branch transition application uses CTMC uniformization:

$$
\exp(Qt) = e^{-\mu t}\sum_{k=0}^{\infty}\frac{(\mu t)^k}{k!}\left(I + \frac{Q}{\mu}\right)^k
$$

All arithmetic is done in `qd_real`.
