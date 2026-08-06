# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.0] - 2026-08-06
### Improvement
- Gradients got approximatly twice as fast for 20 states. For more states the difference is bigger, because they only go in quadratically instead of cubic.

## [2.3.2] - 2026-07-14
### Fix
- A non zero diagonal of S could lead to incorrect gradients, this has now been fixed and the diagonal is ignored like it is documented.

## [2.3.1] - 2026-06-22

### Additions
- update_branch_lengths function for FelsensteinTree

## [2.3.0] - 2026-05-26

### Changes
- For broadcasted parameter like the branch_lengths and S and sqrt_pi (if only one is provided) we used to calculate the gradients with respect to the summed log likelihood of the columns. Now we return the gradients per column.
- We used to give -inf as likelihood if the eigenvalues of the rate matrix where very negative. This is actually numerically stable, so we allow this now.

## [2.2.0] - 2026-03-06

### Additions
- added a function calculate_gradients_with_branch_lengths() which also calculates the gradient of the branch lengths

## [2.1.1] - 2026-02-24

### Fix
- Division by zero for small alignments because of the multithreading strategy in the single model mode

## [2.1.0] - 2026-02-24

### Changes
- Implement Debug and Clone for FelsensteinTree and SingleSideResult
- Report MSRV as 1.87 because of the nalgebra dependecy


## [2.0.0] — 2026-02-18

### Changes

- The input to phylo_grad are now the probabilties at the leaves and not the log probabilties anymore!
- The implementation keeps partial likelihoods in linspace internally. It rescales if nessesacry to avoid under and overflows. This leads to a factor 2 speedup, more for the single model case. The output is still a log probability (natural log)
- We dropped support for f32
- The parallization of the single model case has been changed to be dramatically more memory efficient.
- The project does not relay on a nighlty compiler anymore.


