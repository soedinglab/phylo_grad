# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-

### Changes

- The input to phylo_grad are now the probabilties at the leaves and not the log probabilties anymore!
- The implementation keeps partial likelihoods in linspace internally. It rescales if nessesacry to avoid under and overflows. This leads to a factor 2 speedup, more for the single model case. The output is still a log probability (natural log)
- We dropped support for f32
- The parallization of the single model case has been changed to be dramatically more memory efficient.
- The project does not relay on a nighlty compiler anymore.
