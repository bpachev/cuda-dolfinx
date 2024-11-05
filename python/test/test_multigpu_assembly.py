from test_cuda_assembly import make_test_domain, make_ufl
from mpi4py import MPI
import cudolfinx as cufem
from dolfinx import fem as fe
from dolfinx.fem import petsc as fe_petsc
import numpy as np
from petsc4py import PETSc
import json

def compute_universal_dofmap(mesh, V, res=1000):
    """Map the global array of dofs to unique geometric information

    This is needed to compute maps between DG dofs on meshes with different partitioning schemes
    """
    
    num_local_dofs = V.dofmap.index_map.size_local
    
    c_to_dofs = V.dofmap.map()
    dofs_to_cells = np.zeros(num_local_dofs, dtype=int)
    for i, cell in enumerate(c_to_dofs):
        for dof in cell:
            if dof >= num_local_dofs: continue
            dofs_to_cells[dof] = i 
    dof_coords = V.tabulate_dof_coordinates()[:num_local_dofs]
    cell_coords = mesh.geometry.x[mesh.geometry.dofmap]
    dof_cell_coords = cell_coords[dofs_to_cells]
    dof_coords = mesh.comm.gather(dof_coords, root=0)
    dof_cell_coords = mesh.comm.gather(dof_cell_coords, root=0)
    if mesh.comm.rank == 0:
        dof_coords = (res*np.concat(dof_coords)).astype(int)
        dof_cell_coords = (res*np.concat(dof_cell_coords)).astype(int)
        i = 0
        keys_to_dofs = {}
        keys = []
        for d_coords, d_cell_coords in zip(dof_coords, dof_cell_coords):
            k = (tuple(d_coords.tolist()), tuple(sorted([tuple(arr.tolist()) for arr in d_cell_coords])))
            keys_to_dofs[k] = i
            keys.append(k)
            i += 1

        return keys, keys_to_dofs

    

def compare_parallel_matrices(mat1, mat2):
    """Compare two distributed PETSc matrices
    """

    _, _, data1 = mat1.getValuesCSR()
    _, _, data2 = mat2.getValuesCSR()
    sum1 = MPI.COMM_WORLD.gather(data1.sum(), root=0)
    sum2 = MPI.COMM_WORLD.gather(data2.sum(), root=0)
    if MPI.COMM_WORLD.rank == 0:
        sum1, sum2 = sum(sum1), sum(sum2)
        print(sum1, sum2, np.allclose(sum1, sum2))
        return np.allclose(sum1, sum2)

def compare_parallel_vectors(vec1, vec2):
    """Compare two distributed PETSc vectors
    """

    sum1 = MPI.COMM_WORLD.gather(vec1.array[:].sum(), root=0)
    sum2 = MPI.COMM_WORLD.gather(vec2.array[:].sum(), root=0)
    if MPI.COMM_WORLD.rank == 0:
        sum1, sum2 = sum(sum1), sum(sum2)
        print(sum1, sum2, np.allclose(sum1, sum2))
        return np.allclose(sum1, sum2)

def test_multigpu_assembly():
    """Check assembly operations across multiple GPUs
    """

    domain = make_test_domain()
    regular_ufl = make_ufl()
    ghosted_domain = cufem.ghost_layer_mesh(domain)
    ghosted_ufl = make_ufl(ghosted_domain)
    asm = cufem.CUDAAssembler()
    for form1, form2 in zip(regular_ufl['matrix'], ghosted_ufl['matrix']):
        form1 = fe.form(form1)
        form2 = cufem.form(form2)
        regular_mat = fe_petsc.create_matrix(form1)
        regular_mat.zeroEntries()
        fe_petsc.assemble_matrix(regular_mat, form1, bcs=regular_ufl['bcs'])
        regular_mat.assemble()
        cuda_mat = asm.assemble_matrix(form2, bcs=ghosted_ufl['bcs'])
        cuda_mat.assemble()
        compare_parallel_matrices(regular_mat, cuda_mat.mat)

    for form1, form2 in zip(regular_ufl['vector'], ghosted_ufl['vector']):
        form1 = fe.form(form1)
        form2 = cufem.form(form2)
        regular_vec = fe_petsc.create_vector(form1)
        with regular_vec.localForm() as loc:
            loc.set(0)
        #regular_vec.zeroEntries()
        fe_petsc.assemble_vector(regular_vec, form1)
        regular_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        cuda_vec = asm.assemble_vector(form2)
        good = compare_parallel_vectors(regular_vec, cuda_vec.vector)
        if MPI.COMM_WORLD.size > 1:
            arr1 = MPI.COMM_WORLD.gather(regular_vec.array, root=0)
            arr2 = MPI.COMM_WORLD.gather(cuda_vec.vector.array, root=0)
            dmap2 = compute_universal_dofmap(ghosted_domain, form2.dolfinx_form.function_spaces[-1])
            if MPI.COMM_WORLD.rank == 0 and not good:
              arr1 = np.concat(arr1)
              arr2 = np.concat(arr2)
              with open("dg_vec.json", "r") as fp:
                  expected = json.load(fp)
              
              violations = 0
              for i, k in enumerate(expected["keys"]):
                  k = (tuple(k[0]), tuple(tuple(a) for a in k[1]))
                  correct_value = expected["arr"][i]
                  computed = arr2[i]
                  if not np.allclose(correct_value, computed):
                      violations += 1
                      if violations == 1:
                        print(f"computed[{dmap2[1][k]}]={computed} != expected[{i}]={correct_value}")
              print(f"total violations: {violations} of {len(arr2)}")
              print(f"Sum error: expected={sum(expected['arr'])}, actual={sum(arr2)}")
              break
        elif len(regular_vec.array) > 3e3:
            dmap = compute_universal_dofmap(ghosted_domain, form2.dolfinx_form.function_spaces[-1])
            with open("dg_vec.json", "w") as fp:
                json.dump({"arr": cuda_vec.vector.array.tolist(), "keys": dmap[0]}, fp)


if __name__ == "__main__":
    
    test_multigpu_assembly()
