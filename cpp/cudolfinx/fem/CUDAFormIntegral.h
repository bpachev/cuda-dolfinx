// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAFormConstants.h>
#include <cudolfinx/fem/CUDAFormCoefficients.h>
#include <cudolfinx/mesh/util.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <cudolfinx/la/CUDAMatrix.h>
#include <cudolfinx/la/CUDASeqMatrix.h>
#include <cudolfinx/la/CUDAVector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/utils.h>
#include <cudolfinx/mesh/CUDAMesh.h>
#include <cudolfinx/mesh/CUDAMeshEntities.h>
#include <dolfinx/mesh/Mesh.h>
#include <cuda.h>
#include <map>
#include <string>
#include <vector>

namespace dolfinx {

namespace la {
class CUDAMatrix;
class CUDAVector;
}

namespace fem {

/**
 * `assembly_kernel_type` is used to enumerate different kinds of
 * assembly kernels that may be used to perform assembly for a form
 * integral.
 */
enum assembly_kernel_type
{
    /**
     * An assembly kernel that computes element matrices, sometimes
     * referred to as local assembly, on a CUDA device, whereas the
     * global assembly, that is, scattering element matrices to a
     * global matrix, is performed on the host using PETSc's
     * MatSetValues().
     */
    ASSEMBLY_KERNEL_LOCAL = 0,

    /**
     * An assembly kernel that performs global assembly of a CSR
     * matrix using a binary search within a row to locate a non-zero
     * in the sparse matrix of the bilinear form.
     */
    ASSEMBLY_KERNEL_GLOBAL,

    /**
     * An assembly kernel that uses a lookup table to locate non-zeros
     * in the sparse matrix corresponding to a bilinear form.
     */
    ASSEMBLY_KERNEL_LOOKUP_TABLE,

    /**
     * An assembly kernel that assembles a sparse matrix row-by-row.
     */
    ASSEMBLY_KERNEL_ROWWISE,

    /**
     * A final entry, whose value is equal to the number of enum
     * values.
     */
    NUM_ASSEMBLY_KERNEL_TYPES,
};

using cuda_kern = std::function<void(int*, const char***, const char***,
                     const char**, const char**)>;

/// low-level function to launch a CUDA kernel
CUresult launch_cuda_kernel(
  CUfunction kernel,
  unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
  unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
  unsigned int shared_mem_size_per_thread_block,
  CUstream stream,
  void** kernel_parameters,
  void** extra,
  bool verbose);

/// Launch an assembly kernel with default block/grid sizes
void launch_assembly_kernel(const CUDA::Context& cuda_context, CUfunction kernel, void** kernel_parameters);

/// CUDA C++ code for a CUDA kernel that performs a binary search
std::string cuda_kernel_binary_search(void);

// Function to convert IntegralType to string
std::string to_string(IntegralType integral_type);

std::pair<std::string, std::string> get_cuda_wrapper(
  std::array<std::map<int, std::pair<std::string, std::string>>, 4>& cuda_wrappers,
  IntegralType integral_type,
  int id);

template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void assemble_vector_cell(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  std::int32_t num_active_mesh_entities,
  CUdeviceptr dactive_mesh_entities,
  const dolfinx::mesh::CUDAMesh<U>& mesh,
  const dolfinx::fem::CUDADofMap& dofmap,
  const dolfinx::fem::CUDAFormConstants<T>& constants,
  const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
  dolfinx::la::CUDAVector& cuda_vector,
  bool verbose)
{
  // Mesh vertex coordinates and cells
  std::int32_t num_cells = mesh.num_cells();
  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();

  // Integral constants and coefficients
  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  // Mapping of cellwise to global degrees of freedom and Dirichlet boundary conditions
  int num_dofs_per_cell = dofmap.num_dofs_per_cell();
  CUdeviceptr ddofmap = dofmap.dofs_per_cell();

  // Global vector
  std::int32_t num_values = cuda_vector.num_local_values();
  CUdeviceptr dvalues = cuda_vector.values_write();

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_cells,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_vertices,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_active_mesh_entities,
    &dactive_mesh_entities,
    &num_constant_values,
    &dconstant_values,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_dofs_per_cell,
    &ddofmap,
    &num_values,
    &dvalues};

  launch_assembly_kernel(cuda_context, kernel, kernel_parameters);

  cuda_vector.restore_values_write();
}
//-----------------------------------------------------------------------------
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void assemble_vector_facet(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  std::int32_t num_active_mesh_entities,
  CUdeviceptr dactive_mesh_entities,
  const dolfinx::mesh::CUDAMesh<U>& mesh,
  const dolfinx::fem::CUDADofMap& dofmap,
 // const dolfinx::fem::CUDADirichletBC<T,U>& bc,
  const dolfinx::fem::CUDAFormConstants<T>& constants,
  const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
  dolfinx::la::CUDAVector& cuda_vector,
  bool verbose,
  bool interior)
{
  // Mesh vertex coordinates and cells
  std::int32_t num_cells = mesh.num_cells();
  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  //CUdeviceptr dcell_permutations = mesh.cell_permutations();

  // Integral constants and coefficients
  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  // Mapping of cellwise to global degrees of freedom and Dirichlet boundary conditions
  int num_dofs_per_cell = dofmap.num_dofs_per_cell();
  CUdeviceptr ddofmap = dofmap.dofs_per_cell();

  // Global vector
  std::int32_t num_values = cuda_vector.num_local_values();
  CUdeviceptr dvalues = cuda_vector.values_write();

  std::vector<void*> kernel_parameters;
  kernel_parameters.insert(kernel_parameters.end(), {
    &num_cells,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_vertices,
    &num_coordinates_per_vertex,
    &dvertex_coordinates});

  std::int32_t tdim = mesh.tdim();
  const dolfinx::mesh::CUDAMeshEntities<U>& facets = mesh.mesh_entities()[tdim-1];
  std::int32_t num_mesh_entities_per_cell = facets.num_mesh_entities_per_cell();
  CUdeviceptr dfacet_permutations = facets.mesh_entity_permutations();
  CUdeviceptr coefficient_values_offsets = coefficients.coefficient_values_offsets();
  std::int32_t num_coefficients = coefficients.num_coefficients();

  if (interior) {
    kernel_parameters.insert(kernel_parameters.end(), {
      &num_mesh_entities_per_cell,
      &dfacet_permutations,
      &num_coefficients,
      &coefficient_values_offsets});
  }

  kernel_parameters.insert(kernel_parameters.end(), {
    &num_active_mesh_entities,
    &dactive_mesh_entities,
    &num_constant_values,
    &dconstant_values,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_dofs_per_cell,
    &ddofmap,
    &num_values,
    &dvalues});

  launch_assembly_kernel(cuda_context, kernel, kernel_parameters.data());

  cuda_vector.restore_values_write();
}

//-----------------------------------------------------------------------------
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void lift_bc_cell(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  const dolfinx::mesh::CUDAMesh<U>& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
  const dolfinx::fem::CUDAFormConstants<T>& constants,
  const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
  double scale,
  std::shared_ptr<dolfinx::la::CUDAVector> x0,
  dolfinx::la::CUDAVector& b,
  bool verbose)
{
  std::int32_t num_cells = mesh.num_cells();
  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  //CUdeviceptr dcell_permutations = mesh.cell_permutations();

  int num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
  int num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
  CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();

  CUdeviceptr dbc_markers1 = bc1.dof_markers();
  CUdeviceptr dbc_values1 = bc1.dof_values();

  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  std::int32_t num_columns = dofmap1.num_dofs();
  CUdeviceptr dx0 = (x0) ? x0->values() : NULL;
  std::int32_t num_rows = b.num_local_values();
  CUdeviceptr db = b.values_write();

  // Launch device-side kernel to compute element matrices
  void * kernel_parameters[] = {
    &num_cells,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_constant_values,
    &dconstant_values,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &ddofmap0,
    &ddofmap1,
    &dbc_markers1,
    &dbc_values1,
    &scale,
    &num_columns,
    &num_rows,
    &dx0,
    &db};
  launch_assembly_kernel(cuda_context, kernel, kernel_parameters);

  b.restore_values_write();
  if (x0) x0->restore_values();
}

//-----------------------------------------------------------------------------
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void lift_bc_facet(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  const dolfinx::mesh::CUDAMesh<U>& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
  const dolfinx::fem::CUDAFormConstants<T>& constants,
  const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
  double scale,
  std::shared_ptr<dolfinx::la::CUDAVector> x0,
  dolfinx::la::CUDAVector& b,
  bool verbose,
  std::int32_t num_mesh_entities,
  CUdeviceptr mesh_entities,
  bool interior)
{

  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();

  int num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
  int num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
  CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();

  CUdeviceptr dbc_markers1 = bc1.dof_markers();
  CUdeviceptr dbc_values1 = bc1.dof_values();

  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  std::int32_t num_columns = dofmap1.num_dofs();
  CUdeviceptr dx0 = (x0) ? x0->values() : NULL;
  std::int32_t num_rows = b.num_local_values();
  CUdeviceptr db = b.values_write();

  // Launch device-side kernel to compute element matrices
  std::vector<void*> kernel_parameters;
  kernel_parameters.insert(
    kernel_parameters.end(), {
    &num_mesh_entities,
    &mesh_entities,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates});

  std::int32_t tdim = mesh.tdim();
  const dolfinx::mesh::CUDAMeshEntities<U>& facets = mesh.mesh_entities()[tdim-1];
  std::int32_t num_mesh_entities_per_cell = facets.num_mesh_entities_per_cell();
  CUdeviceptr dfacet_permutations = facets.mesh_entity_permutations();
  CUdeviceptr coefficient_values_offsets = coefficients.coefficient_values_offsets();
  std::int32_t num_coefficients = coefficients.num_coefficients();
 
  if (interior) {
    kernel_parameters.insert(kernel_parameters.end(), {
      &num_mesh_entities_per_cell,
      &dfacet_permutations,
      &num_coefficients,
      &coefficient_values_offsets});
  }

  kernel_parameters.insert(
    kernel_parameters.end(), {
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_constant_values,
    &dconstant_values,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &ddofmap0,
    &ddofmap1,
    &dbc_markers1,
    &dbc_values1,
    &scale,
    &num_columns,
    &num_rows,
    &dx0,
    &db});
  launch_assembly_kernel(cuda_context, kernel, kernel_parameters.data());
  b.restore_values_write();
  if (x0) x0->restore_values();
}

CUDA::Module compile_form_integral_kernel(
  const CUDA::Context& cuda_context,
  CUjit_target target,
  int form_rank,
  IntegralType integral_type,
  std::pair<std::string, std::string> tabulate_tensor_source,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  int32_t num_coeffs_per_cell,
  enum assembly_kernel_type assembly_kernel_type,
  bool debug,
  const char* cudasrcdir,
  bool verbose,
  std::string& factory_name);

/// A wrapper for a form integral with a CUDA-based assembly kernel
/// and data that is stored in the device memory of a CUDA device.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CUDAFormIntegral
{
public:
  /// Create an empty form_integral
  CUDAFormIntegral()
    : _integral_type()
    , _id()
    , _name()
    , _cudasrcdir()
    , _num_vertices_per_cell()
    , _num_coordinates_per_vertex()
    , _num_dofs_per_cell0()
    , _num_dofs_per_cell1()
    , _num_mesh_entities()
    , _mesh_entities()
    , _dmesh_entities(0)
    , _num_mesh_ghost_entities()
    , _mesh_ghost_entities()
    , _dmesh_ghost_entities(0)
    , _element_values()
    , _delement_values(0)
    , _num_nonzero_locations()
    , _dnonzero_locations(0)
    , _delement_matrix_rows(0)
    , _assembly_module()
    , _assembly_kernel_type()
    , _assembly_kernel()
    , _compute_lookup_table_kernel()
    , _lift_bc_kernel()
  {
  }

  /// Create a form_integral
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form_integrals A collection of form integrals
  /// @param[in] integral_type The type of integral
  /// @param[in] i The number of the integral among the integrals of
  ///              the given type belonging to the collection
  /// @param[in] assembly_kernel_type The type of assembly kernel to use
  /// @param[in] cudasrcdir Path for outputting CUDA C++ code
  CUDAFormIntegral(
    const CUDA::Context& cuda_context,
    CUjit_target target,
    const Form<T,U>& form,
    std::pair<std::string, std::string> tabulate_tensor_source,
    IntegralType integral_type, int i,
    int32_t max_threads_per_block,
    int32_t min_blocks_per_multiprocessor,
    int32_t num_vertices_per_cell,
    int32_t num_coordinates_per_vertex,
    int32_t num_dofs_per_cell0,
    int32_t num_dofs_per_cell1,
    enum assembly_kernel_type assembly_kernel_type,
    bool debug,
    const char* cudasrcdir,
    bool verbose)
    : _integral_type(integral_type)
    , _id(i)
    , _cudasrcdir(cudasrcdir)
    , _num_vertices_per_cell()
    , _num_coordinates_per_vertex()
    , _num_dofs_per_cell0(num_dofs_per_cell0)
    , _num_dofs_per_cell1(num_dofs_per_cell1)
    , _num_mesh_entities()
    , _mesh_entities()
    , _dmesh_entities(0)
    , _num_mesh_ghost_entities()
    , _mesh_ghost_entities()
    , _dmesh_ghost_entities(0)
    , _element_values()
    , _delement_values(0)
    , _num_nonzero_locations()
    , _dnonzero_locations(0)
    , _delement_matrix_rows(0)
    , _assembly_module(compile_form_integral_kernel(
                         cuda_context,
                         target,
                         form.rank(),
                         _integral_type,
			 tabulate_tensor_source,
                         max_threads_per_block,
                         min_blocks_per_multiprocessor,
                         num_vertices_per_cell,
                         num_coordinates_per_vertex,
                         num_dofs_per_cell0,
                         num_dofs_per_cell1,
                         form.coefficient_offsets().back(),
                         assembly_kernel_type,
                         debug, cudasrcdir, verbose, _name))
    , _assembly_kernel_type(assembly_kernel_type)
    , _assembly_kernel()
    , _compute_lookup_table_kernel()
    , _lift_bc_kernel()
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    std::string kern_name = std::string("assemble_") + _name;
    switch (integral_type) {
           case IntegralType::cell:
                   kern_name += std::string("_c");
                   break;
           case IntegralType::exterior_facet:
                   kern_name += std::string("_ef");
                   break;
           case IntegralType::interior_facet:
                   kern_name += std::string("_if");
                   break;
    }

    switch (form.rank()) {
      case 0:
        kern_name += std::string("_scalar");
        break;
      case 1:
        kern_name += std::string("_vec");
        break;
      case 2:
      default:
        kern_name += std::string("_mat");
        break;	
    }

    _assembly_kernel = _assembly_module.get_device_function(kern_name);

    if (form.rank() == 2 &&
        (_assembly_kernel_type == ASSEMBLY_KERNEL_LOOKUP_TABLE ||
         _assembly_kernel_type == ASSEMBLY_KERNEL_ROWWISE))
    {
      _compute_lookup_table_kernel = _assembly_module.get_device_function(
        std::string("compute_lookup_table_") +
        std::string("assemble_") + _name);
    }

    if (form.rank() == 2) {
      _lift_bc_kernel = _assembly_module.get_device_function(
        std::string("lift_bc_") + _name);
    }

    if (form.rank() == 0) {
      CUDA::safeMemAlloc(&_dscalar_value, sizeof(T)); 
    }
  #if 0
    // Set the preferred cache configuration to make more shared memory
    // available to the kernel
    cuda_err = cuFuncSetCacheConfig(
      _assembly_kernel, CU_FUNC_CACHE_PREFER_SHARED);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuFuncSetCacheConfig() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  #endif

    // Allocate device-side storage for mesh entities
    _mesh_entities = form.domain(_integral_type, i);
    _num_mesh_entities = _mesh_entities.size();
    _mesh_ghost_entities = mesh::active_ghost_entities(_mesh_entities, _integral_type, form.mesh()->topology_mutable());
    _num_mesh_ghost_entities = _mesh_ghost_entities.size();
    
    if (_num_mesh_entities + _num_mesh_ghost_entities > 0) {
      size_t dmesh_entities_size =
        (_num_mesh_entities + _num_mesh_ghost_entities) * sizeof(int32_t);
      cuda_err = cuMemAlloc(
        &_dmesh_entities, dmesh_entities_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
      _dmesh_ghost_entities = (CUdeviceptr) (
        ((char *) _dmesh_entities) + _num_mesh_entities * sizeof(int32_t));

      // Copy mesh entities to device
      cuda_err = cuMemcpyHtoD(
        _dmesh_entities, _mesh_entities.data(),
        _num_mesh_entities * sizeof(int32_t));
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dmesh_entities);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy mesh ghost entities to device
      if (_num_mesh_ghost_entities > 0) {
        cuda_err = cuMemcpyHtoD(
        _dmesh_ghost_entities, _mesh_ghost_entities.data(),
        _num_mesh_ghost_entities * sizeof(int32_t));
        if (cuda_err != CUDA_SUCCESS) {
          cuGetErrorString(cuda_err, &cuda_err_description);
          cuMemFree(_dmesh_entities);
          throw std::runtime_error(
            "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
            " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
          }
      }
    }

    // Allocate host- and device-side storage for element vector or
    // matrix values
    if (_assembly_kernel_type == dolfinx::fem::ASSEMBLY_KERNEL_LOCAL &&
        (_num_mesh_entities > 0 || _num_mesh_ghost_entities > 0))
    {
      size_t num_element_values =
        (size_t) (_num_mesh_entities + _num_mesh_ghost_entities) *
        _num_dofs_per_cell0 * _num_dofs_per_cell1;
      size_t delement_values_size = num_element_values * sizeof(PetscScalar);
      _element_values = std::vector<PetscScalar>(num_element_values);
      cuda_err = cuMemAlloc(
        &_delement_values, delement_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dmesh_entities);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Destructor
  ~CUDAFormIntegral()
  {
    if (_delement_matrix_rows)
      cuMemFree(_delement_matrix_rows);
    if (_dnonzero_locations)
      cuMemFree(_dnonzero_locations);
    if (_delement_values)
      cuMemFree(_delement_values);
    _element_values.clear();
    if (_dmesh_entities)
      cuMemFree(_dmesh_entities);
    if (_dscalar_value)
      cuMemFree(_dscalar_value);
  }

  //-----------------------------------------------------------------------------
  CUDAFormIntegral(CUDAFormIntegral&& form_integral)
    : _integral_type(form_integral._integral_type)
    , _id(form_integral._id)
    , _name(form_integral._name)
    , _cudasrcdir(form_integral._cudasrcdir)
    , _num_vertices_per_cell(form_integral._num_vertices_per_cell)
    , _num_coordinates_per_vertex(form_integral._num_coordinates_per_vertex)
    , _num_dofs_per_cell0(form_integral._num_dofs_per_cell0)
    , _num_dofs_per_cell1(form_integral._num_dofs_per_cell1)
    , _num_mesh_entities(form_integral._num_mesh_entities)
    , _mesh_entities(form_integral._mesh_entities)
    , _dmesh_entities(form_integral._dmesh_entities)
    , _num_mesh_ghost_entities(form_integral._num_mesh_ghost_entities)
    , _mesh_ghost_entities(form_integral._mesh_ghost_entities)
    , _dmesh_ghost_entities(form_integral._dmesh_ghost_entities)
    , _element_values(form_integral._element_values)
    , _delement_values(form_integral._delement_values)
    , _num_nonzero_locations(form_integral._num_nonzero_locations)
    , _dnonzero_locations(form_integral._dnonzero_locations)
    , _delement_matrix_rows(form_integral._delement_matrix_rows)
    , _assembly_module(std::move(form_integral._assembly_module))
    , _assembly_kernel_type(form_integral._assembly_kernel_type)
    , _assembly_kernel(form_integral._assembly_kernel)
    , _compute_lookup_table_kernel(form_integral._compute_lookup_table_kernel)
    , _lift_bc_kernel(form_integral._lift_bc_kernel)
  {
    form_integral._integral_type = IntegralType::cell;
    form_integral._id = 0;
    form_integral._name = std::string();
    form_integral._cudasrcdir = nullptr;
    form_integral._num_vertices_per_cell = 0;
    form_integral._num_coordinates_per_vertex = 0;
    form_integral._num_dofs_per_cell0 = 0;
    form_integral._num_dofs_per_cell1 = 0;
    form_integral._num_mesh_entities = 0;
    form_integral._dmesh_entities = 0;
    form_integral._num_mesh_ghost_entities = 0;
    form_integral._dmesh_ghost_entities = 0;
    form_integral._element_values = std::vector<PetscScalar>();
    form_integral._delement_values = 0;
    form_integral._num_nonzero_locations = 0;
    form_integral._dnonzero_locations = 0;
    form_integral._delement_matrix_rows = 0;
    form_integral._assembly_module = CUDA::Module();
    form_integral._assembly_kernel_type = ASSEMBLY_KERNEL_GLOBAL;
    form_integral._assembly_kernel = 0;
    form_integral._compute_lookup_table_kernel = 0;
    form_integral._lift_bc_kernel = 0;
  }
  //-----------------------------------------------------------------------------
  CUDAFormIntegral<T,U>& operator=(CUDAFormIntegral<T,U>&& form_integral)
  {
    _integral_type = form_integral._integral_type;
    _id = form_integral._id;
    _name = form_integral._name;
    _cudasrcdir = form_integral._cudasrcdir;
    _num_vertices_per_cell = form_integral._num_vertices_per_cell;
    _num_coordinates_per_vertex = form_integral._num_coordinates_per_vertex;
    _num_dofs_per_cell0 = form_integral._num_dofs_per_cell0;
    _num_dofs_per_cell1 = form_integral._num_dofs_per_cell1;
    _num_mesh_entities = form_integral._num_mesh_entities;
    _mesh_entities = form_integral._mesh_entities;
    _dmesh_entities = form_integral._dmesh_entities;
    _num_mesh_ghost_entities = form_integral._num_mesh_ghost_entities;
    _mesh_ghost_entities = form_integral._mesh_ghost_entities;
    _dmesh_ghost_entities = form_integral._dmesh_ghost_entities;
    _element_values = form_integral._element_values;
    _delement_values = form_integral._delement_values;
    _num_nonzero_locations = form_integral._num_nonzero_locations;
    _dnonzero_locations = form_integral._dnonzero_locations;
    _delement_matrix_rows = form_integral._delement_matrix_rows;
    _assembly_module = std::move(form_integral._assembly_module);
    _assembly_kernel_type = form_integral._assembly_kernel_type;
    _assembly_kernel = form_integral._assembly_kernel;
    _compute_lookup_table_kernel = form_integral._compute_lookup_table_kernel;
    _lift_bc_kernel = form_integral._lift_bc_kernel;
    form_integral._integral_type = IntegralType::cell;
    form_integral._id = 0;
    form_integral._name = std::string();
    form_integral._cudasrcdir = nullptr;
    form_integral._num_vertices_per_cell = 0;
    form_integral._num_coordinates_per_vertex = 0;
    form_integral._num_dofs_per_cell0 = 0;
    form_integral._num_dofs_per_cell1 = 0;
    form_integral._num_mesh_entities = 0;
    form_integral._dmesh_entities = 0;
    form_integral._num_mesh_ghost_entities = 0;
    form_integral._dmesh_ghost_entities = 0;
    form_integral._element_values = std::vector<PetscScalar>();
    form_integral._delement_values = 0;
    form_integral._num_nonzero_locations = 0;
    form_integral._dnonzero_locations = 0;
    form_integral._delement_matrix_rows = 0;
    form_integral._assembly_module = CUDA::Module();
    form_integral._assembly_kernel_type = ASSEMBLY_KERNEL_GLOBAL;
    form_integral._assembly_kernel = 0;
    form_integral._compute_lookup_table_kernel = 0;
    form_integral._lift_bc_kernel = 0;
    return *this;
  }

    /// Get the type of integral
    IntegralType integral_type() const { return _integral_type; }

    /// Get the identifier of the integral
    int id() const { return _id; }

    /// Get the number of mesh entities that the integral applies to
    int32_t num_mesh_entities() const { return _num_mesh_entities; }

    /// Get the mesh entities that the integral applies to
    CUdeviceptr mesh_entities() const { return _dmesh_entities; }

    /// Get the number of mesh ghost entities that the integral applies to
    int32_t num_mesh_ghost_entities() const { return _num_mesh_ghost_entities; }

    /// Get the mesh ghost entities that the integral applies to
    CUdeviceptr mesh_ghost_entities() const { return _dmesh_ghost_entities; }

    /// Get the type of assembly kernel
    enum assembly_kernel_type assembly_kernel_type() const {
        return _assembly_kernel_type; }

    /// Get a handle to the assembly kernel
    CUfunction assembly_kernel() const { return _assembly_kernel; }

  //-----------------------------------------------------------------------------
  /// Assemble a scalar from the form integral
  T assemble_scalar(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    bool verbose) const
  {
    bool interior = false;
    bool facet = false;
  
    switch (_integral_type) {
    case IntegralType::cell:
      break;
    case IntegralType::interior_facet:
      interior = true;
    case IntegralType::exterior_facet:
      facet = true;
      break;
    default:
      throw std::runtime_error(
        "Forms of type " + to_string(_integral_type) + " are not supported "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Mesh vertex coordinates and cells
    std::int32_t num_cells = mesh.num_cells();
    std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
    CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
    std::int32_t num_vertices = mesh.num_vertices();
    std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
    CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();

    // Integral constants and coefficients
    std::int32_t num_constant_values = constants.num_constant_values();
    CUdeviceptr dconstant_values = constants.constant_values();
    std::int32_t num_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

    std::vector<void*> kernel_parameters;
    kernel_parameters.insert(kernel_parameters.end(), {
      &num_cells,
      &num_vertices_per_cell,
      &dvertex_indices_per_cell,
      &num_vertices,
      &num_coordinates_per_vertex,
      &dvertex_coordinates});

    std::int32_t tdim = (facet) ? mesh.tdim()-1 : mesh.tdim();
    const dolfinx::mesh::CUDAMeshEntities<U>& entities = mesh.mesh_entities()[tdim];
    std::int32_t num_mesh_entities_per_cell = entities.num_mesh_entities_per_cell();
    CUdeviceptr dentity_permutations = entities.mesh_entity_permutations();
    CUdeviceptr coefficient_values_offsets = coefficients.coefficient_values_offsets();
    std::int32_t num_coefficients = coefficients.num_coefficients();
    std::int32_t num_active_mesh_entities = _num_mesh_entities + _num_mesh_ghost_entities;
    CUdeviceptr active_mesh_entities = _dmesh_entities;
    CUdeviceptr value = _dscalar_value;

    if (interior) {
      kernel_parameters.insert(kernel_parameters.end(), {
        &num_mesh_entities_per_cell,
        &dentity_permutations,
        &num_coefficients,
        &coefficient_values_offsets});
    }

    kernel_parameters.insert(kernel_parameters.end(), {
      &num_active_mesh_entities,
      &active_mesh_entities,
      &num_constant_values,
      &dconstant_values,
      &num_coefficient_values_per_cell,
      &dcoefficient_values,
      &value});

    // reset accumulator to zero
    set_scalar_value(0.0);
    launch_assembly_kernel(cuda_context, _assembly_kernel, kernel_parameters.data());

    return get_scalar_value();
  }
  //-----------------------------------------------------------------------------
  /// Assemble a vector from the form integral
  void assemble_vector(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap,
   // const dolfinx::fem::CUDADirichletBC<T,U>& bc,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAVector& cuda_vector,
    bool verbose) const
  {
    CUresult cuda_err;
    const char * cuda_err_description;
    bool interior = false;
    switch (_integral_type) {
    case IntegralType::cell:
      assemble_vector_cell(
        cuda_context, _assembly_kernel, _num_mesh_entities + _num_mesh_ghost_entities, _dmesh_entities,
        mesh, dofmap, constants, coefficients, cuda_vector, verbose);
      break;
    case IntegralType::interior_facet:
      interior = true;
    case IntegralType::exterior_facet:
      assemble_vector_facet(
        cuda_context, _assembly_kernel, _num_mesh_entities + _num_mesh_ghost_entities, _dmesh_entities,
        mesh, dofmap, constants, coefficients, cuda_vector, verbose, interior);
      break;
    default:
      throw std::runtime_error(
        "Forms of type " + to_string(_integral_type) + " are not supported "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
  //-----------------------------------------------------------------------------
  /// Assemble a matrix from the form integral
  void lift_bc(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    double scale,
    std::shared_ptr<dolfinx::la::CUDAVector> x0,
    dolfinx::la::CUDAVector& b,
    bool verbose) const
  {
    CUresult cuda_err;
    const char * cuda_err_description;
    bool interior = false;
    switch (_integral_type) {
    case IntegralType::cell:
      lift_bc_cell(
        cuda_context, _lift_bc_kernel, mesh, dofmap0, dofmap1,
        bc1, constants, coefficients, scale, x0, b, verbose);
      break;
    case IntegralType::interior_facet:
      interior = true;
    case IntegralType::exterior_facet:
      lift_bc_facet(
        cuda_context, _lift_bc_kernel, mesh, dofmap0, dofmap1,
        bc1, constants, coefficients, scale, x0, b, verbose,
        _num_mesh_entities + _num_mesh_ghost_entities, _dmesh_entities,
        interior);
      break;
    default:
      throw std::runtime_error(
        "Forms of type " + to_string(_integral_type) + " are not supported "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
  //-----------------------------------------------------------------------------
  /// Assemble a matrix from the form integral performing local
  /// assembly on a CUDA device and global assembly on the host
  void assemble_matrix_local(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    CUfunction kernel = _assembly_kernel;
    std::int32_t num_mesh_entities = _num_mesh_entities;
    CUdeviceptr mesh_entities = _dmesh_entities;
    CUdeviceptr element_values = _delement_values;

    std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
    CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
    std::int32_t num_vertices = mesh.num_vertices();
    std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
    CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
    //CUdeviceptr dcell_permutations = mesh.cell_permutations();

    std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
    std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
    CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();
    CUdeviceptr dcells_per_dof_ptr = dofmap0.cells_per_dof_ptr();
    CUdeviceptr dcells_per_dof = dofmap0.cells_per_dof();

    CUdeviceptr dbc0 = bc0.dof_markers();
    CUdeviceptr dbc1 = bc1.dof_markers();

    CUdeviceptr dconstant_values = constants.constant_values();
    std::int32_t num_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

    CUdeviceptr dvalues = _delement_values;

    // Use the CUDA occupancy calculator to determine a grid and block
    // size for the CUDA kernel
    int min_grid_size;
    int block_size;
    int shared_mem_size_per_thread_block = 0;
    cuda_err = cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, 0, 0, 0);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuOccupancyMaxPotentialBlockSize() failed with " +
        std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    unsigned int grid_dim_x = min_grid_size;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = block_size;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    CUstream stream = NULL;

    // Launch device-side kernel to compute element matrices
    (void) cuda_context;
    void * kernel_parameters[] = {
      &num_mesh_entities,
      &mesh_entities,
      &num_vertices_per_cell,
      &dvertex_indices_per_cell,
      &num_coordinates_per_vertex,
      &dvertex_coordinates,
      &num_coefficient_values_per_cell,
      &dcoefficient_values,
      &dconstant_values,
     // &dcell_permutations,
      &num_dofs_per_cell0,
      &num_dofs_per_cell1,
      &ddofmap0,
      &ddofmap1,
      &dbc0,
      &dbc1,
      &dvalues};
    cuda_err = launch_cuda_kernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters, NULL, verbose);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    // Wait for the kernel to finish.
    cuda_err = cuCtxSynchronize();
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }
  //-----------------------------------------------------------------------------
  /// Copy element matrices from a CUDA device and perform global
  /// assembly on the host
  void assemble_matrix_local_copy_to_host(
    const CUDA::Context& cuda_context)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Since the kernel only performed local assembly, we must copy
    // element matrices from the device to the host and perform the
    // global assembly on the host.
    cuda_err = cuMemcpyDtoH(
      _element_values.data(), _delement_values,
      _element_values.size() * sizeof(PetscScalar));
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
  //-----------------------------------------------------------------------------
  /// Perform global assembly on the host
  void assemble_matrix_local_host_global_assembly(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    dolfinx::la::CUDAMatrix& A)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();

    PetscErrorCode ierr;
    Mat host_A = A.mat();
    const dolfinx::fem::DofMap& host_dofmap0 = *dofmap0.dofmap();
    const dolfinx::fem::DofMap& host_dofmap1 = *dofmap1.dofmap();
    for (int i = 0; i < _num_mesh_entities; i++) {
      std::int32_t c = _mesh_entities[i];
      auto element_dofs0 = host_dofmap0.cell_dofs(c);
      auto element_dofs1 = host_dofmap1.cell_dofs(c);
      const PetscScalar* Ae = &_element_values[
        i * num_dofs_per_cell0 * num_dofs_per_cell1];
      ierr = MatSetValuesLocal(
        host_A, _num_dofs_per_cell0, element_dofs0.data(),
        _num_dofs_per_cell1, element_dofs1.data(), Ae, ADD_VALUES);
      if (ierr != 0)
        dolfinx::la::petsc::error(ierr, __FILE__, "MatSetValuesLocal");
    }
  }
  //-----------------------------------------------------------------------------
  void assemble_matrix_global(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    CUfunction kernel = _assembly_kernel;
    std::int32_t num_mesh_entities =
      _num_mesh_entities + _num_mesh_ghost_entities;
    CUdeviceptr mesh_entities = _dmesh_entities;
    CUdeviceptr element_values = _delement_values;

    std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
    CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
    std::int32_t num_vertices = mesh.num_vertices();
    std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
    CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
    //CUdeviceptr dcell_permutations = mesh.cell_permutations();

    std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
    std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
    CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();

    CUdeviceptr dbc0 = bc0.dof_markers();
    CUdeviceptr dbc1 = bc1.dof_markers();

    CUdeviceptr dconstant_values = constants.constant_values();
    std::int32_t num_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

    std::int32_t num_local_rows = A.num_local_rows();
    std::int32_t num_local_columns = A.num_local_columns();
    CUdeviceptr drow_ptr = A.diag()->row_ptr();
    CUdeviceptr dcolumn_indices = A.diag()->column_indices();
    CUdeviceptr dvalues = A.diag()->values();
    CUdeviceptr doffdiag_row_ptr = A.offdiag() ? A.offdiag()->row_ptr() : 0;
    CUdeviceptr doffdiag_column_indices = A.offdiag() ? A.offdiag()->column_indices() : 0;
    CUdeviceptr doffdiag_values = A.offdiag() ? A.offdiag()->values() : 0;
    std::int32_t num_local_offdiag_columns = A.num_local_offdiag_columns();
    CUdeviceptr dcolmap_sorted = A.colmap_sorted();
    CUdeviceptr dcolmap_sorted_indices = A.colmap_sorted_indices();

    // Use the CUDA occupancy calculator to determine a grid and block
    // size for the CUDA kernel
    int min_grid_size;
    int block_size;
    int shared_mem_size_per_thread_block = 0;
    cuda_err = cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, 0, 0, 0);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuOccupancyMaxPotentialBlockSize() failed with " +
        std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    unsigned int grid_dim_x = min_grid_size;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = block_size;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    CUstream stream = NULL;

    // Launch device-side kernel to compute element matrices and
    // perform global assembly
    (void) cuda_context;
    void * kernel_parameters[] = {
      &num_mesh_entities,
      &mesh_entities,
      &num_vertices_per_cell,
      &dvertex_indices_per_cell,
      &num_coordinates_per_vertex,
      &dvertex_coordinates,
      &num_coefficient_values_per_cell,
      &dcoefficient_values,
      &dconstant_values,
      //&dcell_permutations,
      &num_dofs_per_cell0,
      &num_dofs_per_cell1,
      &ddofmap0,
      &ddofmap1,
      &dbc0,
      &dbc1,
      &num_local_rows,
      &num_local_columns,
      &drow_ptr,
      &dcolumn_indices,
      &dvalues,
      &doffdiag_row_ptr,
      &doffdiag_column_indices,
      &doffdiag_values,
      &num_local_offdiag_columns,
      &dcolmap_sorted,
      &dcolmap_sorted_indices
    };
    cuda_err = launch_cuda_kernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters, NULL, verbose);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    // Wait for the kernel to finish.
    cuda_err = cuCtxSynchronize();
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

  }
  //-----------------------------------------------------------------------------
  void assemble_matrix_lookup_table(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Compute a lookup table, unless it has been computed already.
    if (!_dnonzero_locations) {
      compute_lookup_table(
        cuda_context, dofmap0, dofmap1,
        bc0, bc1, A, verbose);
    }

    CUfunction kernel = _assembly_kernel;
    std::int32_t num_mesh_entities = _num_mesh_entities;
    CUdeviceptr mesh_entities = _dmesh_entities;
    CUdeviceptr nonzero_locations = _dnonzero_locations;

    std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
    CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
    std::int32_t num_vertices = mesh.num_vertices();
    std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
    CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
    //CUdeviceptr dcell_permutations = mesh.cell_permutations();

    std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();

    CUdeviceptr dconstant_values = constants.constant_values();
    std::int32_t num_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

    /* TODO: Pass in the diagonal and off-diagonal part of the matrix. */
    CUdeviceptr dvalues = A.diag()->values();

    // Use the CUDA occupancy calculator to determine a grid and block
    // size for the CUDA kernel
    int min_grid_size;
    int block_size;
    int shared_mem_size_per_thread_block = 0;
    cuda_err = cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, 0, 0, 0);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuOccupancyMaxPotentialBlockSize() failed with " +
        std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    unsigned int grid_dim_x = min_grid_size;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = block_size;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    CUstream stream = NULL;

    // Launch device-side kernel to compute element matrices
    (void) cuda_context;
    void * kernel_parameters[] = {
      &num_mesh_entities,
      &mesh_entities,
      &num_vertices_per_cell,
      &dvertex_indices_per_cell,
      &num_coordinates_per_vertex,
      &dvertex_coordinates,
      &num_coefficient_values_per_cell,
      &dcoefficient_values,
      &dconstant_values,
      //&dcell_permutations,
      &num_dofs_per_cell0,
      &num_dofs_per_cell1,
      &nonzero_locations,
      &dvalues};
    cuda_err = launch_cuda_kernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters, NULL, verbose);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    // Wait for the kernel to finish.
    cuda_err = cuCtxSynchronize();
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }
  //-----------------------------------------------------------------------------
  void assemble_matrix_rowwise(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Compute a lookup table, unless it has been computed already.
    if (!_dnonzero_locations) {
      compute_lookup_table(
        cuda_context, dofmap0, dofmap1,
        bc0, bc1, A, verbose);
    }

    CUfunction kernel = _assembly_kernel;
    std::int32_t num_mesh_entities = _num_mesh_entities;
    CUdeviceptr mesh_entities = _dmesh_entities;
    CUdeviceptr nonzero_locations = _dnonzero_locations;
    CUdeviceptr element_matrix_rows = _delement_matrix_rows;

    std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
    CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
    std::int32_t num_vertices = mesh.num_vertices();
    std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
    CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
    //CUdeviceptr dcell_permutations = mesh.cell_permutations();

    std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
    CUdeviceptr dcells_per_dof_ptr = dofmap0.cells_per_dof_ptr();
    CUdeviceptr dcells_per_dof = dofmap0.cells_per_dof();

    CUdeviceptr dconstant_values = constants.constant_values();
    std::int32_t num_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

    /* TODO: Pass in the diagonal and off-diagonal part of the matrix. */
    std::int32_t num_rows = A.num_rows();
    CUdeviceptr dvalues = A.diag()->values();

    // Use the CUDA occupancy calculator to determine a grid and block
    // size for the CUDA kernel
    int min_grid_size;
    int block_size;
    int shared_mem_size_per_thread_block = 0;
    cuda_err = cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, 0, 0, 0);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuOccupancyMaxPotentialBlockSize() failed with " +
        std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    unsigned int grid_dim_x = min_grid_size;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = block_size;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    CUstream stream = NULL;

    // Launch device-side kernel to compute element matrices
    (void) cuda_context;
    void * kernel_parameters[] = {
      &num_mesh_entities,
      &mesh_entities,
      &num_vertices_per_cell,
      &dvertex_indices_per_cell,
      &num_coordinates_per_vertex,
      &dvertex_coordinates,
      &num_coefficient_values_per_cell,
      &dcoefficient_values,
      &dconstant_values,
     // &dcell_permutations,
      &num_dofs_per_cell0,
      &num_dofs_per_cell1,
      &dcells_per_dof_ptr,
      &dcells_per_dof,
      &nonzero_locations,
      &element_matrix_rows,
      &num_rows,
      &dvalues};
    cuda_err = launch_cuda_kernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters, NULL, verbose);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    // Wait for the kernel to finish.
    cuda_err = cuCtxSynchronize();
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }

  void assemble_matrix_facet(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose,
    bool interior
    )
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    CUfunction kernel = _assembly_kernel;
    std::int32_t num_mesh_entities =
      _num_mesh_entities + _num_mesh_ghost_entities;
    CUdeviceptr mesh_entities = _dmesh_entities;
    CUdeviceptr element_values = _delement_values;
    std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
    CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
    std::int32_t num_vertices = mesh.num_vertices();
    std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
    CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
    //CUdeviceptr dcell_permutations = mesh.cell_permutations();
    std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
    std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
    CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();
    CUdeviceptr dbc0 = bc0.dof_markers();
    CUdeviceptr dbc1 = bc1.dof_markers();
    CUdeviceptr dconstant_values = constants.constant_values();
    std::int32_t num_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

    std::int32_t num_local_rows = A.num_local_rows();
    std::int32_t num_local_columns = A.num_local_columns();
    CUdeviceptr drow_ptr = A.diag()->row_ptr();
    CUdeviceptr dcolumn_indices = A.diag()->column_indices();
    CUdeviceptr dvalues = A.diag()->values();
    CUdeviceptr doffdiag_row_ptr = A.offdiag() ? A.offdiag()->row_ptr() : 0;
    CUdeviceptr doffdiag_column_indices = A.offdiag() ? A.offdiag()->column_indices() : 0;
    CUdeviceptr doffdiag_values = A.offdiag() ? A.offdiag()->values() : 0;
    std::int32_t num_local_offdiag_columns = A.num_local_offdiag_columns();
    CUdeviceptr dcolmap_sorted = A.colmap_sorted();
    CUdeviceptr dcolmap_sorted_indices = A.colmap_sorted_indices();


    // Use the CUDA occupancy calculator to determine a grid and block
    // size for the CUDA kernel
    int min_grid_size;
    int block_size;
    int shared_mem_size_per_thread_block = 0;
    cuda_err = cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, 0, 0, 0);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuOccupancyMaxPotentialBlockSize() failed with " +
        std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    unsigned int grid_dim_x = min_grid_size;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = block_size;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    CUstream stream = NULL;

    // Launch device-side kernel to compute element matrices and
    // perform global assembly
    (void) cuda_context;
    std::vector<void*> kernel_parameters;
    kernel_parameters.insert(kernel_parameters.end(), {
      &num_mesh_entities,
      &mesh_entities,
      &num_vertices_per_cell,
      &dvertex_indices_per_cell,
      &num_coordinates_per_vertex,
      &dvertex_coordinates});
    if (interior)
    {
      std::int32_t tdim = mesh.tdim();
      const dolfinx::mesh::CUDAMeshEntities<U>& facets = mesh.mesh_entities()[tdim-1];
      std::int32_t num_mesh_entities_per_cell = facets.num_mesh_entities_per_cell();
      CUdeviceptr dfacet_permutations = facets.mesh_entity_permutations();
      CUdeviceptr coefficient_values_offsets = coefficients.coefficient_values_offsets();
      std::int32_t num_coefficients = coefficients.num_coefficients();
      kernel_parameters.insert(kernel_parameters.end(), {
      &num_mesh_entities_per_cell,
      &dfacet_permutations,
      &num_coefficients,
      &coefficient_values_offsets});
    }
    kernel_parameters.insert(kernel_parameters.end(), {
      &num_coefficient_values_per_cell,
      &dcoefficient_values,
      &dconstant_values,
      &num_dofs_per_cell0,
      &num_dofs_per_cell1,
      &ddofmap0,
      &ddofmap1,
      &dbc0,
      &dbc1,
      &num_local_rows,
      &num_local_columns,
      &drow_ptr,
      &dcolumn_indices,
      &dvalues,
      &doffdiag_row_ptr,
      &doffdiag_column_indices,
      &doffdiag_values,
      &num_local_offdiag_columns,
      &dcolmap_sorted,
      &dcolmap_sorted_indices});


    cuda_err = launch_cuda_kernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters.data(), NULL, verbose);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    // Wait for the kernel to finish.
    cuda_err = cuCtxSynchronize();
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

  }

  //-----------------------------------------------------------------------------
  /// Assemble a matrix from the form integral
  void assemble_matrix(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose)
  {

    // special handling for facet integrals
    bool interior = false;
    switch (_integral_type) {
    case IntegralType::interior_facet:
      interior = true;
    case IntegralType::exterior_facet:
      return assemble_matrix_facet(
        cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
        constants, coefficients, A, verbose, interior
      );
    default:
      break;
    }

    switch (_assembly_kernel_type) {
    case ASSEMBLY_KERNEL_LOCAL:
      return assemble_matrix_local(
        cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
        constants, coefficients, A, verbose);
    case ASSEMBLY_KERNEL_GLOBAL:
      return assemble_matrix_global(
        cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
        constants, coefficients, A, verbose);
    case ASSEMBLY_KERNEL_LOOKUP_TABLE:
      return assemble_matrix_lookup_table(
        cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
        constants, coefficients, A, verbose);
    case ASSEMBLY_KERNEL_ROWWISE:
      return assemble_matrix_rowwise(
        cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
        constants, coefficients, A, verbose);
    default:
      throw std::invalid_argument(
        "Invalid argument at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
  //-----------------------------------------------------------------------------
  /// Compute the lookup table that is used during matrix assembly for
  /// kernels that need it
  void compute_lookup_table(
      const CUDA::Context& cuda_context,
      const dolfinx::fem::CUDADofMap& dofmap0,
      const dolfinx::fem::CUDADofMap& dofmap1,
      const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
      const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
      dolfinx::la::CUDAMatrix& A,
      bool verbose)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    if (!_compute_lookup_table_kernel)
      return;

    int num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
    int num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();

    // Get the warp size
    int warp_size;
    cuda_err = cuDeviceGetAttribute(
      &warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuda_context.device());
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuDeviceGetAttribute() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Allocate device-side storage for the lookup table for the sparse
    // matrix non-zeros, rounding the number of mesh cells up to the
    // nearest multiple of the warp size due to coalescing.
    _num_nonzero_locations =
      ((int64_t) ((_num_mesh_entities + (warp_size-1)) / warp_size) * warp_size) *
      num_dofs_per_cell0 * num_dofs_per_cell1;
    if (_num_nonzero_locations <= 0)
      return;

    size_t dnonzero_locations_size = _num_nonzero_locations * sizeof(int32_t);
    cuda_err = cuMemAlloc(&_dnonzero_locations, dnonzero_locations_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    size_t delement_matrix_rows_size =
      (int64_t) _num_mesh_entities * _num_dofs_per_cell0 * sizeof(int32_t);
    cuda_err = cuMemAlloc(&_delement_matrix_rows, delement_matrix_rows_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    CUfunction kernel = _compute_lookup_table_kernel;
    std::int32_t num_mesh_entities = _num_mesh_entities;
    CUdeviceptr mesh_entities = _dmesh_entities;

    std::int32_t num_dofs = dofmap0.num_dofs();
    CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
    CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();
    CUdeviceptr dcells_per_dof_ptr = dofmap0.cells_per_dof_ptr();
    CUdeviceptr dcells_per_dof = dofmap0.cells_per_dof();

    CUdeviceptr dbc0 = bc0.dof_markers();
    CUdeviceptr dbc1 = bc1.dof_markers();

    /* TODO: Pass in the diagonal and off-diagonal part of the matrix. */
    std::int32_t num_rows = A.num_rows();
    CUdeviceptr drow_ptr = A.diag()->row_ptr();
    CUdeviceptr dcolumn_indices = A.diag()->column_indices();

    // Use the CUDA occupancy calculator to determine a grid and block
    // size for the CUDA kernel
    int min_grid_size;
    int block_size;
    int shared_mem_size_per_thread_block = 0;
    cuda_err = cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, 0, 0, 0);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuOccupancyMaxPotentialBlockSize() failed with " +
        std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    unsigned int grid_dim_x = min_grid_size;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = block_size;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    CUstream stream = NULL;

    // Launch device-side kernel to compute element matrices
    (void) cuda_context;
    void * kernel_parameters[] = {
      &num_mesh_entities,
      &mesh_entities,
      &num_dofs,
      &num_dofs_per_cell0,
      &num_dofs_per_cell1,
      &ddofmap0,
      &ddofmap1,
      &dcells_per_dof_ptr,
      &dcells_per_dof,
      &dbc0,
      &dbc1,
      &num_rows,
      &drow_ptr,
      &dcolumn_indices,
      &_num_nonzero_locations,
      &_dnonzero_locations,
      &_delement_matrix_rows};
    cuda_err = launch_cuda_kernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters, NULL, verbose);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    // Wait for the kernel to finish.
    cuda_err = cuCtxSynchronize();
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }

  /// Copy constructor
  /// @param[in] form_integral The object to be copied
  CUDAFormIntegral(const CUDAFormIntegral& form_integral) = delete;


  /// Assignment operator
  /// @param[in] form_integral Another CUDAFormIntegral object
  CUDAFormIntegral& operator=(const CUDAFormIntegral& form_integral) = delete;

  /// Setter for device value for scalar integrals
  /// @param[in] value Value to set
  void set_scalar_value(T value) const {
    CUDA::safeMemcpyHtoD(_dscalar_value, &value, sizeof(T));
  }

  /// Getter for device value for scalar integrals
  T get_scalar_value() const {
    T value = 0.0;
    CUDA::safeMemcpyDtoH(&value, _dscalar_value, sizeof(T));
    return value;
  }

private:
  /// Type of the integral
  IntegralType _integral_type;

  /// Identifier for the integral
  int _id;

  /// A name for the integral assigned to it by UFC
  std::string _name;

  /// Path to use for outputting CUDA C++ code
  const char* _cudasrcdir;

  /// The number of vertices in a mesh cell
  int32_t _num_vertices_per_cell;

  /// The number of coordinates for each vertex
  int32_t _num_coordinates_per_vertex;

  /// The number of degrees of freedom per cell associated with the
  /// first function space
  int32_t _num_dofs_per_cell0;

  /// The number of degrees of freedom per cell associated with the
  /// second function space
  int32_t _num_dofs_per_cell1;

  /// The number of mesh entities that the integral applies to
  int32_t _num_mesh_entities;

  /// Host-side storage for mesh entities that the integral applies to
  std::span<const std::int32_t> _mesh_entities;

  /// Device-side storage for mesh entities that the integral applies to
  CUdeviceptr _dmesh_entities;

  /// The number of mesh ghost entities that the integral applies to
  int32_t _num_mesh_ghost_entities;

  /// Host-side storage for mesh ghost entities that the integral applies to
  std::vector<std::int32_t> _mesh_ghost_entities;

  /// Device-side storage for mesh ghost entities that the integral applies to
  CUdeviceptr _dmesh_ghost_entities;

  /// Host-side storage for element vector or matrix values, which is
  /// used for kernels that only perform local assembly on a CUDA
  /// device, but perform global assembly on the host.
  std::vector<PetscScalar> _element_values;

  /// Device-side storage for element vector or matrix values, which
  /// is used for kernels that only perform local assembly on a CUDA
  /// device, but perform global assembly on the host.
  CUdeviceptr _delement_values;

  /// The number of entries in the lookup table
  int64_t _num_nonzero_locations;

  /// A lookup table that is used to find the locations of non-zeros
  /// in the sparse matrix for the degrees of freedom of each mesh
  /// entity.
  CUdeviceptr _dnonzero_locations;

  /// Another lookup table that is used during rowwise assembly which
  /// maps each mesh cell containing a degree of freedom of the test
  /// space to the corresponding row of the computed element matrix.
  CUdeviceptr _delement_matrix_rows;

  /// CUDA module contaitning compiled and loaded device code for
  /// assembly kernels based on the form integral
  CUDA::Module _assembly_module;

  /// The type of assembly kernel to use
  enum assembly_kernel_type _assembly_kernel_type;

  /// CUDA kernel for assembly based on the form integral
  CUfunction _assembly_kernel;

  /// CUDA kernel for computing a lookup table for assembling the form integral
  CUfunction _compute_lookup_table_kernel;

  /// CUDA kernel for imposing essential boundary conditions
  CUfunction _lift_bc_kernel;

  /// CUDA scalar value for scalar integrals
  CUdeviceptr _dscalar_value;
};


//-----------------------------------------------------------------------------
/// Compile assembly kernels for a collection of form integrals and
/// copy the necessary data to a CUDA device
///
/// @param[in] cuda_context A context for a CUDA device
/// @param[in] form A variational form
/// @param[in] assembly_kernel_type The type of assembly kernel to use
/// @param[in] cudasrcdir Path for outputting CUDA C++ code
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>
cuda_form_integrals(
  const CUDA::Context& cuda_context,
  CUjit_target target,
  const Form<T,U>& form,
  std::array<std::map<int, std::pair<std::string, std::string>>, 4>& cuda_wrappers,
  enum assembly_kernel_type assembly_kernel_type,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  bool debug,
  const char* cudasrcdir,
  bool verbose)
{
  // there is no integrals function
  //const FormIntegrals& form_integrals = form.integrals();

  // Get the number of vertices and coordinates
  const mesh::Mesh<U>& mesh = *form.mesh();
  std::int32_t num_vertices_per_cell = mesh::num_cell_vertices(mesh.geometry().cmap().cell_shape());
  //std::int32_t num_coordinates_per_vertex = mesh.geometry().dim();
  std::int32_t num_coordinates_per_vertex = 3;

  // Find the number of degrees of freedom per cell
  int32_t num_dofs_per_cell0 = 1;
  int32_t num_dofs_per_cell1 = 1;
  if (form.rank() > 0) {
    const DofMap& dofmap0 = *form.function_spaces()[0]->dofmap();
    num_dofs_per_cell0 = dofmap0.element_dof_layout().num_dofs() * dofmap0.element_dof_layout().block_size();
  }
  if (form.rank() > 1) {
    const DofMap& dofmap1 = *form.function_spaces()[1]->dofmap();
    num_dofs_per_cell1 = dofmap1.element_dof_layout().num_dofs() * dofmap1.element_dof_layout().block_size();
  }
  std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>
    cuda_form_integrals;

  {
    // Create device-side kernels and data for cell integrals
    IntegralType integral_type = IntegralType::cell;
    int num_integrals = form.num_integrals(integral_type);
    if (num_integrals > 0) {
      std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals =
        cuda_form_integrals[integral_type];
      for (int i : form.integral_ids(integral_type)) {

        cuda_cell_integrals.emplace_back(
          cuda_context, target, form, get_cuda_wrapper(cuda_wrappers, integral_type, i), integral_type, i,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0, num_dofs_per_cell1,
          assembly_kernel_type, debug, cudasrcdir, verbose);
      }
    }
  }

  {
    // Create device-side kernels and data for exterior facet integrals
    IntegralType integral_type = IntegralType::exterior_facet;
    int num_integrals = form.num_integrals(integral_type);
    if (num_integrals > 0) {
      std::vector<CUDAFormIntegral<T,U>>& cuda_exterior_facet_integrals =
        cuda_form_integrals[integral_type];
      for (int i : form.integral_ids(integral_type)) {
        cuda_exterior_facet_integrals.emplace_back(
          cuda_context, target, form, get_cuda_wrapper(cuda_wrappers, integral_type, i), integral_type, i,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0, num_dofs_per_cell1,
          assembly_kernel_type, debug, cudasrcdir, verbose);
      }
    }
  }

  {
    // Create device-side kernels and data for interior facet integrals
    IntegralType integral_type = IntegralType::interior_facet;
    int num_integrals = form.num_integrals(integral_type);
    if (num_integrals > 0) {
      std::vector<CUDAFormIntegral<T,U>>& cuda_interior_facet_integrals =
        cuda_form_integrals[integral_type];
      for (int i : form.integral_ids(integral_type)) {
        cuda_interior_facet_integrals.emplace_back(
          cuda_context, target, form, get_cuda_wrapper(cuda_wrappers, integral_type, i), integral_type, i,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0, num_dofs_per_cell1,
          assembly_kernel_type, debug, cudasrcdir, verbose);
      }
    }
  }

  return cuda_form_integrals;
}


} // namespace fem
} // namespace dolfinx
