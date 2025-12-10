# Efficient and Parallel gradients of substitution models for Phylogenetic trees

## Repository structure

`benchmark_test` contains python code to benchmark on random data and test the gradients against a pytorch implementation. It also contains a snakemake pipeline to generate the plots in the paper.

`phylo_grad` contains the Rust code for use in Rust without any python dependecies.

`phylo_grad_py` contains python bindings for `phylo_grad`.

`phylo_grad_gpu` contains a GPU implementation using jax.

`gtr_optimize` contains the Rust optimization code used in the RaxML/IQ-TREE benchmark of the paper.

## Using in Python

We provide a pip package `phylo_grad`, it contains precompiled wheels for python 3.11, 3.12, 3.13 and 3.14 for manylinux_2_28 on x86_64 with AVX2. If you need to install on another plattform you can still do it with pip, but you need to fullfill the requirements of the Compiling from source section. This means having a rust toolchain, cmake, gcc, gfortran and the environment variables set.

## Compiling from source
You need a working Rust compiler, the easiest is to install rustup : https://www.rust-lang.org/tools/install
We depend on a specific version of the compiler for now to get better performance, rustup will download the correct toolchain for you if you compile from this repository.

You also have to have `cmake`, `gcc`, and `gfortran` available on the system.

It is recommended to install it into a conda environment, it needs at least python 3.7

 To install the package into your current environment, run
 ```
 conda create -n phylo_grad python=3.11
 source activate phylo_grad
 # Disable mulithreading inside of openblas (this can lead to problems and does not give better performance)
 export USE_LOCKING=1
 export USE_THREAD=0
 export USE_OPENMP=0
 # target-cpu=native can improve performance on the machine it is compiled for
 export RUSTFLAGS="-C target-cpu=native"
 pip install ./phylo_grad_py
 ```

## Using from Rust

Just `cargo add phylo_grad`.

For compiling you need `cmake`, `gcc` and `gfortran` available on the system.
We strongly recommend to set these environment variables during compilation:
```
 export USE_LOCKING=1
 export USE_THREAD=0
 export USE_OPENMP=0
```
For more detail see `phylo_grad/Readme.md`.

## Running benchmarks/tests

Create and activate a conda environment:

```
conda env create -n phylo_grad -c conda-forge python=3.11 
source activate phylo_grad
```

Install phylo_grad (requires rustup):

```
pip install ./phylo_grad_py
pip install ./phylo_grad_gpu
```

Install other dependecies:
```
conda install -c conda-forge pytorch bioconda::snakemake pytest bioconda::newick_utils bioconda::iqtree bioconda::emboss
cargo install phylotree
```

Run Tests

```
cd benchmark_test
pytest test.py
```

Run Benchmarks

```
snakemake -c all data/random/time_t={num_threads}_L={columns in alignment}_m={method}.pickle
```

`method` can be either `rust`, `pytorch`, `pytorch_gpu` or `jax_gpu`.

## How to use it

See `examples/train.py` for a simple example usage. There is a pydoc for the complete API, for example accessable with `python -m pydoc phylo_grad` after installation.

## Trouble Shooting

### Unsupported DIM in Python

If you use the python bindings you can by default only use 4, 16 and 20 for the number of states in the Felsenstein.
If you need other numbers you can add them at the end of `phylo_grad_py/src/lib.rs` (needs a change in 2 positions) and compile from source.
