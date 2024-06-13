# Setup

This repository contains examples of GPU acceleration with FEniCSx. In order to run the examples, you will need to build FEniCSx from source, using my forks of ffcx and dolfinx. The differences in the build process are as follows:

- the flag `-DDOLFINX_ENABLE_CUDATOOLKIT=ON` must be passed when configuring dolfinx.
- PETSc needs to be compiled with GPU support enabled. 

## v0.8.0 (recommended)

For a version compatible with dolfinx 0.8.0, clone the branches version-0.8.0 of my dolfinx and ffcx forks

```
git clone --branch version-0.8.0 git@github.com:bpachev/ffcx
git clone --branch version-0.8.0 git@github.com:bpachev/dolfinx
```

## main

For access to the current development version, clone the main branches

```
git clone git@github.com:bpachev/ffcx
git clone git@github.com:bpachev/dolfinx
```
