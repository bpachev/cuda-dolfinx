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

This repository currently relies on a from source build of dolfinx 0.9.0. PETSc also needs to be built with CUDA support enabled.

# Installation

The installation process is set up to mirror the dolfinx library, with a C++ component installed via CMake, and a Python package installed with pip. For help with installing or using the library, feel free to contact me at benjaminpachev@gmail.com.
