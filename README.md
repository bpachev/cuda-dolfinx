# About

This repository is an add-on extension to the DOLFINx library providing CUDA accelerated assembly routines. It complements the CUDA linear solvers in PETSc to enable fully GPU-accelerated DOLFINx codes. It is designed to enable GPU acceleration for existing codes with minimal changes.

# Basic Usage

```
import cudolfinx as cufem

# given UFL forms A and L representing a stiffness matrix and right-hand-side
cuda_A = cufem.form(A)
cuda_L = cufem.form(L)
asm = cufem.CUDAAssembler()
# returns a custom type CUDAMatrix
mat = asm.assemble_matrix(cuda_A)
mat.assemble()
# get PETSc matrix
petsc_mat = mat.mat()
# returns a custom type CUDAVector
vec = asm.assemble_vector(cuda_L)
#get PETSc vector
petsc_vec = vec.vector()
```

# Dependencies

- dolfinx 0.10.0
- PETSc with CUDA support
- CUDA Toolkit 12.x

# Installation

There are three ways to do the install, in increasing order of difficulty. Currently, it is not possible to use `cudolfinx` with the existing Conda and Docker distributions of `dolfinx`, because these force installation of PETSc without CUDA support. Consequently, installing `cudolfinx` requires a custom modification to the `dolfinx` dependency stack that has CUDA-enabled PETSc.

## Docker

Using Docker is by far the easiest approach.

```
docker run --gpus all -it benpachev/cudolfinx:v0.10.0-cuda12.6
```
You may experience errors with the prebuilt container due to CUDA Toolkit or MPI version mismatch between the host and container. In this case, the Dockerfiles in `docker/` can be modified to use a different CUDA Toolkit version or MPI version to build a container that will work with your system. Note that on HPC systems, Docker is not available, but Docker containers can be converted to Apptainer/Singularity containers.

```
apptainer pull docker://benpachev/cudolfinx:v0.10.0-cuda12.6
apptainer run --nv cudolfinx_v0.10.0-cuda12.6.sif
```

## Spack

Spack is a management tool for HPC software, and allows for an extreme amount of flexibility in compilation of code and dependency selection. It has somewhat of a learning curve, and typically doesn't work out of the box without some manual configuration. However, it can be a good choice for HPC systems without Apptainer installed, or if more control over the compilation process and dependencies is desired. To install with Spack:

```
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
spack env create cudolfinx-env
spacktivate cudolfinx-env
git clone https://github.com/bpachev/cuda-dolfinx.git
spack repo add cuda-dolfinx/spack
spack add cuda-dolfinx py-cuda-dolfinx
spack install
```

If this leads to errors, it is likely due to either (a) Spack is unable to find a suitable compiler or properly configure your existing compiler (b) Spack is trying to build a poorly supported low-level package from source. To resolve (a), you can usually do `spack compiler add`. Especially on HPC systems, [additional configuration](https://spack-tutorial.readthedocs.io/en/latest/tutorial_configuration.html#compiler-configuration) is usually needed. To solve (b), you will often need to [force Spack to use existing](https://spack-tutorial.readthedocs.io/en/latest/tutorial_configuration.html#external-packages) low-level software on your system instead of trying to install it from source.

## From Source

The difficult part about the install is the dependencies. The Dockerfiles under `docker/` provide a template for how to install the dependencies on Debian-based systems (and using Docker is by far the easiest way to get a development environment). Once that is taken care of, the installation of `cuda-dolfinx` itself is simple. 

### C++ Core
```
cd cpp
mkdir build
cmake ..  -DCUDOLFINX_SKIP_BUILD_TESTS=YES
make install
```

### Python Bindings:
```
cd python
pip --check-build-dependencies --no-build-isolation .
```

For help with installing or using the library, feel free to contact me at benjaminpachev@gmail.com.
