# PhyloGrad

This is the Rust implementation of PhyloGrad. For more information go to the main repository https://github.com/soedinglab/phylo_grad

## Lapack

We use lapack for matrix diagonalization. This is the only part which is not implemented in Rust and the most likely cause of problems. The default will use a static `openblas` build, this turned out to work best on most systems, but it takes a while to compile. `openblas` will by default use multithreading which is not desirable for `phylo_grad` and can even cause unsoundness because we call into `openblas` from `rayon`. With setting the environment variables
```
 export USE_LOCKING=1
 export USE_THREAD=0
 export USE_OPENMP=0
```
during compilation we can tell `openblas` to not use multithreading.