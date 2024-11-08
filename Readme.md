# Efficient and Parallel gradients of substitution models for Phylogenetic trees

## Repository structure

`benchmark_test` contains python code to benchmark on random data and test the gradients against a pytorch implementation. It also contains a snakemake pipeline to generate the plots in the paper.

`phylo_grad` contains the pure Rust code usage from Rust without any python dependencies

`phylo_grad_py` contains python bindings for `phylo_grad`

## Using from Python
You need a working Rust compiler, the easiest is to install rustup : https://www.rust-lang.org/tools/install
We depend on a specific version of the compiler for now to get better performance, rustup will download the correct toolchain for you if you compile from this repository.

You also have to have `cmake`, `gcc`, and `gfortran` available on the system.

It is recommended to install it into a conda environment, it needs at least python 3.7

 To install the package into your current environment, run
 ```
 conda create -n phylo_grad python=3.11
 source activate phylo_grad
 export RUSTFLAGS="-C target-cpu=native"
 pip install ./phylo_grad_py
 ```
 
## Using from Rust

Just do `cargo add phylo_grad` (will work after publishing)

## Running benchmarks/tests

Create and activate a conda environment:

```
conda env create -n phylo_grad --file=benchmark_test/environment.yml
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
snakemake -c all data/random/plot_{num_threads}_time.pdf
```
