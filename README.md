# About

NOTICE: This repository is still under construction - examples and documentation are forthcoming.

This repository is an add-on extension to the DOLFINx library providing GPU accelerated assembly routines. Currently only NVIDIA GPUs are supported, and only single-GPU assembly is supported.

# Setup

This repository depends on my custom forks of dolfinx and ffcx. The dolfinx library must be built from source:

- the flag `-DDOLFINX_ENABLE_CUDATOOLKIT=ON` must be passed when configuring dolfinx.
- PETSc needs to be compiled with GPU support enabled. 

## Custom branches

 The following branches need to be cloned of my forks:

```
git clone --branch version-0.8.0 git@github.com:bpachev/ffcx
git clone --branch standalone-package git@github.com:bpachev/dolfinx
```

Versions of basix and ufl also need to be cloned that are compatible with dolfinx 0.8.0.


