# Efficient and Parallel gradients of substitution models for Phylogenetic trees

## Repository structure

`benchmark_test` contains python code to benchmark on random data and test the gradients against a pytorch implementation. It also contains a snakemake pipeline to generate the plots in the paper.

`phylo_grad` contains the pure Rust code usage from Rust without any python dependencies

`phylo_grad_py` contains python bindings for `phylo_grad`

`phylo_grad_gpu` contains a GPU implementation using jax

## Using from Python
You need a working Rust compiler, the easiest is to install rustup : https://www.rust-lang.org/tools/install
We depend on a specific version of the compiler for now to get better performance, rustup will download the correct toolchain for you if you compile from this repository.

You also have to have `cmake`, `gcc`, and `gfortran` available on the system.

It is recommended to install it into a conda environment, it needs at least python 3.7

 To install the package into your current environment, run
 ```
 conda create -n phylo_grad python=3.11
 source activate phylo_grad
 export NUM_THREADS="256"
 export RUSTFLAGS="-C target-cpu=native"
 pip install ./phylo_grad_py
 ```
`NUM_THREADS` governs the maximum amount of threads that openblas will be able to use. If not set it will use the number of cores of the machine where it is compiled, which can be a problem on a HPC cluster environment.

After publication there will be a precompiled pip package for easy use.

## Using from Rust

Just `cargo add phylo_grad` (will work after publishing)

## Running benchmarks/tests

Create and activate a conda environment:

```
conda env create -n phylo_grad python=3.11 pytorch bioconda::snakemake pytest bioconda::newick_utils
source activate phylo_grad
```

Install phylo_grad (requires rustup):

```
pip install ./phylo_grad_py
```

Run Tests

```
cd benchmark_test
pytest test.py
```

Install phylotree (to generate random trees)

```
cargo install phylotree
```

Run Benchmarks

```
snakemake -c all data/random/time_t={num_threads}_L={columns in alignment}_m={method}.pickle"
```

`method` can be either `rust`, `pytorch`, `pytorch_gpu` or `jax_gpu`.

## How to use it

See `examples/train.py` for a simple example usage. There is a pydoc for the complete API, for example accessable with `python -m pydoc phylo_grad` after installation.

## Trouble Shooting

### Unsupported DIM in Python

If you use the python bindings you can by default only use 4, 16 and 20 for the number of states in the Felsenstein.
If you need other numbers you can add them at the end of `phylo_grad_py/src/lib.rs` (needs a change in 2 positions) and compile from source.
