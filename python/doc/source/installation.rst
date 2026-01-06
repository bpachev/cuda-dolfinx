
Installation
============

cuDOLFINx can be run through containers, installed with Spack, or built from source. Binary distributions are planned, but are currently unavailable. This is partially due to the fact that existing distributions of DOLFINx force installation of PETSc without CUDA support, while cuDOLFINx works best with CUDA-enabled PETSc.

Docker/Apptainer
----------------

Using a container is by far the easiest approach::

    docker run --gpus all -it benpachev/cudolfinx:v0.9.0-cuda12.6

You may experience errors with the prebuilt container due to CUDA Toolkit or MPI version mismatch between the host and container. In this case, the `Dockerfiles <https://github.com/bpachev/cuda-dolfinx/tree/main/docker>`_ bundled with the source code can be modified to use a different CUDA Toolkit version or MPI version to build a container that will work with your system. Note that on HPC systems, Docker is not available, but Docker containers can be converted to Apptainer/Singularity containers::


    apptainer pull docker://benpachev/cudolfinx:v0.9.0-cuda12.6
    apptainer run --nv cudolfinx_v0.9.0-cuda12.6.sif


Spack
-----

`Spack <https://spack.io/>`_ is a management tool for HPC software, and allows for an extreme amount of flexibility in compilation of code and dependency selection. It has somewhat of a learning curve, and typically doesn't work out of the box without some manual configuration. However, it can be a good choice for HPC systems without Apptainer installed, or if more control over the compilation process and dependencies is desired. To install with Spack::

    git clone https://github.com/spack/spack.git
    . spack/share/spack/setup-env.sh
    spack env create cudolfinx-env
    spacktivate cudolfinx-env
    git clone https://github.com/bpachev/cuda-dolfinx.git
    spack repo add cuda-dolfinx/spack
    spack add cuda-dolfinx py-cuda-dolfinx
    spack install

If this leads to errors, it is likely due to either (a) Spack is unable to find a suitable compiler or properly configure your existing compiler (b) Spack is trying to build a poorly supported low-level package from source. To resolve (a), you can usually do `spack compiler add`. Especially on HPC systems, `additional configuration <https://spack-tutorial.readthedocs.io/en/latest/tutorial_configuration.html#compiler-configuration>`_ is usually needed. To solve (b), you will often need to `force Spack to use existing <https://spack-tutorial.readthedocs.io/en/latest/tutorial_configuration.html#external-packages>`_ low-level software on your system instead of trying to install it from source.

Source
------

Like DOLFINx, cuDOLFINx has a C++ core and a Python interface. While it is possible to use only the C++ core, it is much simpler to work with Python interface.

Dependencies
^^^^^^^^^^^^

The only dependencies are DOLFINx, the CUDA Toolkit, and PETSc with CUDA support. cuDOLFINx only provides GPU accelerated assembly operations - PETSC is needed for GPU accelerated linear solves.


C++
***

After obtaining the source, the C++ core can be installed with cmake::

    cd cpp
    mkdir build
    cmake ..  -DCUDOLFINX_SKIP_BUILD_TESTS=YES
    make install


Python
******

Once the C++ core is installed, the Python interface is installed with::

    cd python
    pip --check-build-dependencies --no-build-isolation .

For help with installing or using the library, feel free to contact me at benjaminpachev@gmail.com.
