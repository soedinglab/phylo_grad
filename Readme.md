# Efficient and Parallel gradients of substitution models for Phylogenetic trees

## Repository structure

`benchmark_test` contains python code to benchmark on random data and test the gradients against a pytorch implementation. It also contains a snakemake pipeline to generate the plots in the paper.

`phylo_grad` contains the pure Rust code usage from Rust without any python dependencies

`phylo_grad_py` contains python bindings for `phylo_grad`

## Compile from Source Python
You need a working Rust compiler, the easiest is to install rustup : https://www.rust-lang.org/tools/install
We depend on a specific version of the compiler for now to get better performance, rustup will download the correct toolchain for you if you compile from this repository.

It is recommended to install it into a conda environment. You have to install `maturin` and `patchelf`.

 To install the package into your current environment, run
 ```
 maturin develop --release
 ```
 Alternatively, you can build the python wheel for any version of python currently installed in your system:
 ```
 maturin build --release
 ```

## Installation Rust

You can just add `felsenstein_impl = "0.1.0"` as a dependency