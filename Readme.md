# Efficient and Parallel gradients of substitution models for Phylogenetic trees

## Repository structure

`benchmark_test` contains python code to benchmark on random data and test the gradients against a pytorch implementation. It also contains a snakemake pipeline to generate the plots in the paper.

`phylo_grad` contains the pure Rust code usage from Rust without any python dependencies

`phylo_grad_py` contains python bindings for `phylo_grad`

## Using from Python
You need a working Rust compiler, the easiest is to install rustup : https://www.rust-lang.org/tools/install
We depend on a specific version of the compiler for now to get better performance, rustup will download the correct toolchain for you if you compile from this repository.

It is recommended to install it into a conda environment.

 To install the package into your current environment, run
 ```
 export RUSTFLAGS="-C target-cpu=native"
 cd phylo_grad_py
 pip install .
 ```
 
## Using from Rust

Just do `cargo add phylo_grad`