import argparse as ap
from mpi4py import MPI
from petsc4py import PETSc
try:
    import cudolfinx as cufem
except ImportError:
    print("Must have cudolfinx to test CUDA assembly.")

from dolfinx import fem as fe, mesh
from dolfinx.fem import petsc as fe_petsc
import numpy as np
import ufl
import time
from ufl import dx, ds, grad, inner 
import basix

def create_mesh(res: int = 10):
    """Create a uniform tetrahedral mesh on the unit cube.

    Parameters
    ----------
    res - Number of subdivisions along each dimension

    Returns
    ----------
    mesh - The mesh object.
    """

    return mesh.create_box(
            comm = MPI.COMM_WORLD,
            points = ((0,0,0), (1, 1, 1)),
            n = (res, res, res),
            cell_type = mesh.CellType.hexahedron,
            ghost_mode = mesh.GhostMode.none,
            dtype = np.float64
        )

def main(res, cuda=True, sum_factorization=True, degree=1):
    """Assembles a stiffness matrix for the Poisson problem with the given resolution.
    """

    domain = create_mesh(res)
    # Tensor product element
    family = basix.ElementFamily.P
    variant = basix.LagrangeVariant.gll_warped
    cell_type = domain.basix_cell()

    basix_element = basix.create_tp_element(
        family, cell_type, degree, variant
    )  # doesn't work with tp element, why?
    element = basix.ufl._BasixElement(basix_element)  # basix ufl element
    V = fe.functionspace(domain, element)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = 10*ufl.exp(-((x[0]-.5)**2 + (x[1]-.5)**2 + (x[2]-.5)**2) / .02)
    g = ufl.sin(5*x[0])*ufl.sin(5*x[1])
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx + inner(g, v) * ds

    facets = mesh.locate_entities_boundary(
      domain,
      dim=(domain.topology.dim - 1),
      marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
    )

    dofs = fe.locate_dofs_topological(V=V, entity_dim=domain.topology.dim-1, entities=facets)
    bc = fe.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V)

    form_compiler_options = {"sum_factorization": sum_factorization}  

    if cuda:
        a = cufem.form(a, form_compiler_options=form_compiler_options)
        asm = cufem.CUDAAssembler()
        A = asm.create_matrix(a)
        device_bcs = asm.pack_bcs([bc])
    else:
        a = fe.form(
                a,
                form_compiler_options=form_compiler_options,
                jit_options = {"cffi_extra_compile_args":["-O3", "-mcpu=neoverse-v2"]}
        )
        A = fe_petsc.create_matrix(a)

    start = time.time()
    if cuda:
        asm.assemble_matrix(a, A, bcs=device_bcs)
    else:
        fe_petsc.assemble_matrix(A, a, bcs=[bc])
        A.assemble()
    elapsed = time.time()-start

    timing = MPI.COMM_WORLD.gather(elapsed, root=0)
    if MPI.COMM_WORLD.rank == 0:
        timing = np.asarray(timing)
        timing = np.max(timing)
        # show max over all MPI processes, as that's the rate-limiter
        print(f"Res={res}, Num cells", domain.topology.index_map(domain.topology.dim).size_global)
        print(f"Assembly timing: {timing}, Dofs: {V.dofmap.index_map.size_global}")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--res", default=10, type=int, help="Number of subdivisions in each dimension.")
    parser.add_argument("--degree", default=1, type=int, help="Polynomial degree.")
    parser.add_argument("--no-sum-factorization", default=False, action="store_true", help="Disable sum factorization")
    parser.add_argument("--no-cuda", default=False, action="store_true", help="Disable GPU acceleration.")
    args = parser.parse_args()

    main(
         res = args.res,
         cuda = not args.no_cuda,
         sum_factorization = not args.no_sum_factorization,
         degree = args.degree
    )
