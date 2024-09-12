// Copyright (C) 2020-2024 James D. Trotter, Benjamin Pachev
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/la/petsc.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <ufcx.h>
#include <cuda.h>
#include <cudolfinx/fem/CUDAFormIntegral.h>
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAFormCoefficients.h>
#include <cudolfinx/fem/CUDAFormConstants.h>
#include <cudolfinx/la/CUDAMatrix.h>
#include <cudolfinx/la/CUDASeqMatrix.h>
#include <cudolfinx/la/CUDAVector.h>
#include <cudolfinx/mesh/CUDAMesh.h>
#include <cudolfinx/mesh/CUDAMeshEntities.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace {

// declarations of functions for generating code snippets
// for use in multiple kernels
std::string get_interior_facet_joint_dofmaps(
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1
);

std::string get_interior_facet_joint_dofmap(
  int32_t num_dofs_per_cell
);

std::string compute_interior_facet_tensor(
 std::string tabulate_tensor_function_name,
 int32_t num_vertices_per_cell,
 int32_t num_coordinates_per_vertex,
 int32_t num_coeffs_per_cell,
 bool vector=false
);

// add debugging code to assembly kernel
std::string dump_assembly_vars(IntegralType integral_type);
std::string dump_arr(const std::string& name, const std::string& length, const std::string& fmt);

std::string interior_facet_extra_args();
std::string interior_facet_pack_cell_coeffs(int32_t num_coeffs_per_cell);

/// CUDA C++ code for cellwise assembly of a vector from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_vector_cell(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_vertices,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
   // "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_dofs_per_cell,\n"
    "  const int32_t* __restrict__ dofmap,\n"
  //  "  const char* __restrict__ bc,\n"
    "  int32_t num_values,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell == " + std::to_string(num_dofs_per_cell) + ");\n"
    "  ufc_scalar_t xe[" + std::to_string(num_dofs_per_cell) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      xe[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
   // "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element vector\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      xe,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation);\n"

    "\n"
    "    // Add element vector values to the global vector,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs = &dofmap[c*" + std::to_string(num_dofs_per_cell) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      int32_t row = dofs[j];\n"
    "      atomicAdd(&values[row], xe[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a vector from a form
/// integral over exterior mesh facets
std::string cuda_kernel_assemble_vector_exterior_facet(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_vertices,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int32_t num_active_mesh_entities,\n"
    "  const int32_t* __restrict__ active_mesh_entities,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_dofs_per_cell,\n"
    "  const int32_t* __restrict__ dofmap,\n"
    //"  const char* __restrict__ bc,\n"
    "  int32_t num_values,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell == " + std::to_string(num_dofs_per_cell) + ");\n"
    "  ufc_scalar_t xe[" + std::to_string(num_dofs_per_cell) + "];\n"
    "\n"
    "  for (int i = 2*thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += 2*blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_mesh_entities[i];\n"
    "    int32_t local_mesh_entity = active_mesh_entities[i+1];\n"
    "    // Set element vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      xe[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
   // "    const uint8_t* quadrature_permutation =\n"
   // "      &mesh_entity_permutations[\n"
   // "        local_mesh_entity*num_cells+c];\n"
   // "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element vector\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      xe,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      &local_mesh_entity,\n"
    "      nullptr);\n"
    "\n"
    "    // Add element vector values to the global vector,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs = &dofmap[c*" + std::to_string(num_dofs_per_cell) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      int32_t row = dofs[j];\n"
    "      atomicAdd(&values[row], xe[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a vector from a form
/// integral over exterior mesh facets
std::string cuda_kernel_assemble_vector_interior_facet(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell,
  int32_t num_coeffs_per_cell)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_vertices,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    + interior_facet_extra_args() +
    "  int32_t num_active_mesh_entities,\n"
    "  const int32_t* __restrict__ active_mesh_entities,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_dofs_per_cell,\n"
    "  const int32_t* __restrict__ dofmap,\n"
    "  int32_t num_values,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[2*" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell == " + std::to_string(num_dofs_per_cell) + ");\n"
    "  ufc_scalar_t xe[2*" + std::to_string(num_dofs_per_cell) + "];\n"
    "\n"
    "  for (int i = 4*thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += 4*blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c0 = active_mesh_entities[i];\n"
    "    int32_t c1 = active_mesh_entities[i+2];\n"
    "    int32_t facet0 = active_mesh_entities[i+1];\n"
    "    int32_t facet1 = active_mesh_entities[i+3];\n"
    "    // Set element vector values to zero\n"
    "    for (int j = 0; j < 2*" + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      xe[j] = 0.0;\n"
    "    }\n"
    "\n"
    + compute_interior_facet_tensor(
       tabulate_tensor_function_name,
       num_vertices_per_cell,
       num_coordinates_per_vertex,
       num_coeffs_per_cell,
       true // set flag indicating vector instead of matrix
    )
    + get_interior_facet_joint_dofmap(num_dofs_per_cell) +
    "\n"
    "    // Add element vector values to the global vector,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    for (int j = 0; j < 2*" + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      int32_t row = dofs[j];\n"
    "      atomicAdd(&values[row], xe[j]);\n"
    "    }\n"
    "  }\n" 
    "}";
}


/// CUDA C++ code for assembly of a vector from a form integral
std::string cuda_kernel_assemble_vector(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  IntegralType integral_type,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell,
  int32_t num_coeffs_per_cell)
{
  switch (integral_type) {
  case IntegralType::cell:
    return cuda_kernel_assemble_vector_cell(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell);
  case IntegralType::exterior_facet:
    return cuda_kernel_assemble_vector_exterior_facet(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell);
  case IntegralType::interior_facet:
    return cuda_kernel_assemble_vector_interior_facet(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell,
      num_coeffs_per_cell);
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

/// CUDA C++ code for modifying a right-hand side vector to impose
/// essential boundary conditions for integrals over mesh cells
std::string cuda_kernel_lift_bc_cell(
  std::string lift_bc_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + lift_bc_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc_markers1,\n"
    "  const ufc_scalar_t* __restrict__ bc_values1,\n"
    "  double scale,\n"
    "  int32_t num_columns,\n"
    "  const ufc_scalar_t* x0,\n"
    "  ufc_scalar_t* b)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "  ufc_scalar_t be[" + std::to_string(num_dofs_per_cell0) + "];\n"
    "\n"
    "  for (int c = thread_idx;\n"
    "    c < num_cells;\n"
    "    c += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    // Skip cell if boundary conditions do not apply\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    bool has_bc = false;\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        has_bc = true;\n"
    "        break;\n"
    "      }\n"
    "    }\n"
    "    if (!has_bc)\n"
    "      continue;\n"
    "\n"
    "    // Set element matrix and vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "      be[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation);\n"
    "\n"
    "    // Compute modified element vector\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        const ufc_scalar_t _x0 = (x0) ? x0[column] : 0.0;\n"
    "        ufc_scalar_t bc = bc_values1[column];\n"
    "        for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "          be[j] -= Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] * scale * (bc - _x0);\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Add element vector values to the global vector\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      atomicAdd(&b[row], be[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for modifying a right-hand side vector to impose
/// essential boundary conditions for integrals over exterior facets
std::string cuda_kernel_lift_bc_exterior_facet(
  std::string lift_bc_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + lift_bc_kernel_name + "(\n"
    "  int32_t num_active_mesh_entities,\n"
    "  int32_t* active_mesh_entities,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc_markers1,\n"
    "  const ufc_scalar_t* __restrict__ bc_values1,\n"
    "  double scale,\n"
    "  int32_t num_columns,\n"
    "  const ufc_scalar_t* x0,\n"
    "  ufc_scalar_t* b)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "  ufc_scalar_t be[" + std::to_string(num_dofs_per_cell0) + "];\n"
    "\n"
    "  for (int i = 2*thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += 2*blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_mesh_entities[i];\n"
    "    int32_t local_mesh_entity = active_mesh_entities[i+1];\n"
    "    // Skip cell if boundary conditions do not apply\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    bool has_bc = false;\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        has_bc = true;\n"
    "        break;\n"
    "      }\n"
    "    }\n"
    "    if (!has_bc)\n"
    "      continue;\n"
    "\n"
    "    // Set element matrix and vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "      be[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      &local_mesh_entity,\n"
    "      quadrature_permutation);\n"
    "\n"
    "    // Compute modified element vector\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        ufc_scalar_t bc = bc_values1[column];\n"
    "        const ufc_scalar_t _x0 = (x0) ? x0[column] : 0.0;\n"
    "        for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "          be[j] -= Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] * scale * (bc - _x0);\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Add element vector values to the global vector\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      atomicAdd(&b[row], be[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for modifying a right-hand side vector to impose
/// essential boundary conditions for integrals over exterior facets
std::string cuda_kernel_lift_bc_interior_facet(
  std::string lift_bc_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  int32_t num_coeffs_per_cell
)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + lift_bc_kernel_name + "(\n"
    "  int32_t num_active_mesh_entities,\n"
    "  int32_t* active_mesh_entities,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    + interior_facet_extra_args() +
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc_markers1,\n"
    "  const ufc_scalar_t* __restrict__ bc_values1,\n"
    "  double scale,\n"
    "  int32_t num_columns,\n"
    "  const ufc_scalar_t* x0,\n"
    "  ufc_scalar_t* b)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[4*" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "  ufc_scalar_t be[2*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "\n"
    "  for (int i = 4*thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += 4*blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c0 = active_mesh_entities[i];\n"
    "    int32_t c1 = active_mesh_entities[i+2];\n"
    "    int32_t facet0 = active_mesh_entities[i+1];\n"
    "    int32_t facet1 = active_mesh_entities[i+3];\n"
    "    // Skip cell if boundary conditions do not apply\n"
    "    const int32_t* dofs10 = &dofmap1[c0*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    const int32_t* dofs11 = &dofmap1[c1*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    bool has_bc = false;\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs10[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        has_bc = true;\n"
    "        break;\n"
    "      }\n"
    "    }\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs11[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        has_bc = true;\n"
    "        break;\n"
    "      }\n"
    "    }\n"
    "    if (!has_bc)\n"
    "      continue;\n"
    "\n"
    + compute_interior_facet_tensor(
       tabulate_tensor_function_name,
       num_vertices_per_cell,
       num_coordinates_per_vertex,
       num_coeffs_per_cell
    )
    + get_interior_facet_joint_dofmaps(num_dofs_per_cell0, num_dofs_per_cell1) +
    "\n"
    "    // Compute modified element vector\n"
    "    for (int k = 0; k < 2*" + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        ufc_scalar_t bc = bc_values1[column];\n"
    "        const ufc_scalar_t _x0 = (x0) ? x0[column] : 0.0;\n"
    "        for (int j = 0; j < 2*" + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "          be[j] -= Ae[j*2*" + std::to_string(num_dofs_per_cell1) + "+k] * scale * (bc - _x0);\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Add element vector values to the global vector\n"
    "    for (int j = 0; j < 2*" + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      atomicAdd(&b[row], be[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for assembly of a matrix from a form integral
std::string cuda_kernel_lift_bc(
  std::string kernel_name,
  std::string tabulate_tensor_function_name,
  IntegralType integral_type,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  int32_t num_coeffs_per_cell)
{
  switch (integral_type) {
  case IntegralType::cell:
    return cuda_kernel_lift_bc_cell(
      kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case IntegralType::exterior_facet:
    return cuda_kernel_lift_bc_exterior_facet(
      kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case IntegralType::interior_facet:
    return cuda_kernel_lift_bc_interior_facet(
      kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      num_coeffs_per_cell);
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

/// CUDA C++ code for cellwise, local assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_local(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
   // "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    ufc_scalar_t* Ae = &values[\n"
    "      i*" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
   // "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation);\n"
    "\n"
    "    // For degrees of freedom that are subject to essential boundary conditions,\n"
    "    // set the element matrix values to zero.\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "        }\n"
    "        continue;\n"
    "      }\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        int32_t column = dofs1[k];\n"
    "        if (bc1 && bc1[column]) {;\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // This kernel does not perform any global assembly. That is,\n"
    "    // only element matrices are computed here. The element matrices\n"
    "    // should be copied to the host, and the host is responsible for\n"
    "    // scattering them to the correct locations in the global matrix.\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_global(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return ""
    "extern \"C\" int printf(const char * format, ...);\n"
    "\n"
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    //"  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_local_rows,\n"
    "  int32_t num_local_columns,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  ufc_scalar_t* __restrict__ values,\n"
    "  const int32_t* __restrict__ offdiag_row_ptr,\n"
    "  const int32_t* __restrict__ offdiag_column_indices,\n"
    "  ufc_scalar_t* __restrict__ offdiag_values,\n"
    "  int32_t num_local_offdiag_columns,\n"
    "  const int32_t* __restrict__ colmap)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    //"    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation);\n"
    "\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) continue;\n"
    "      if (row < num_local_rows) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) continue;\n"
    "          if (column < num_local_columns) {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[row+1] - row_ptr[row],\n"
    "              &column_indices[row_ptr[row]],\n"
    "              column, &r);\n"
    "            assert(!err && \"Failed to find column index in assemble_matrix_cell_global!\");\n"
    "            r += row_ptr[row];\n"
    "            atomicAdd(&values[r],\n"
    "              Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          } else {\n"
    "            /* Search for the correct column index in the column map\n"
    "             * of the off-diagonal part of the local matrix. */\n"
    "            int32_t colmap_idx = -1;\n"
    "            for (int q = 0; q < num_local_offdiag_columns; q++) {\n"
    "              if (column == colmap[q]) {\n"
    "                colmap_idx = q;\n"
    "                break;\n"
    "              }\n"
    "            }\n"
    "            assert(colmap_idx != -1);\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              offdiag_row_ptr[row+1] - offdiag_row_ptr[row],\n"
    "              &offdiag_column_indices[offdiag_row_ptr[row]],\n"
    "              colmap_idx, &r);\n"
    "            assert(!err && \"Failed to find offdiag column index in assemble_matrix_cell_global!\");\n"
    "            r += offdiag_row_ptr[row];\n"
    "            atomicAdd(&offdiag_values[r],\n"
    "              Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for computing a lookup table for the sparse matrix
/// non-zeros corresponding to the degrees of freedom of each mesh entity
std::string cuda_kernel_compute_lookup_table(
  std::string assembly_kernel_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    "extern \"C\" void __global__\n"
    "compute_lookup_table_" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_dofs,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const int32_t* __restrict__ cells_per_dof_ptr,\n"
    "  const int32_t* __restrict__ cells_per_dof,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_rows,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  int64_t num_nonzero_locations,\n"
    "  int32_t* __restrict__ nonzero_locations,\n"
    "  int32_t* __restrict__ element_matrix_rows)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    // Use a binary search to locate non-zeros in the sparse matrix.\n"
    "    // For degrees of freedom that are subject to essential boundary\n"
    "    // conditions, insert a negative value in the lookup table.\n"
    "    int32_t c = active_cells[i];\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (((int64_t) (i / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell0) + " + (int64_t) j) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + (int64_t) k) *\n"
    "            warpSize + (i % warpSize);\n"
    "            nonzero_locations[l] = -1;\n"
    "        }\n"
    "      } else {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (((int64_t) (i / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell0) + " + (int64_t) j) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + (int64_t) k) *\n"
    "            warpSize + (i % warpSize);\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) {\n"
    "            nonzero_locations[l] = -1;\n"
    "          } else {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[row+1] - row_ptr[row],\n"
    "              &column_indices[row_ptr[row]],\n"
    "              column, &r);\n"
    "            nonzero_locations[l] = row_ptr[row] + r;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_lookup_table(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    cuda_kernel_compute_lookup_table(
      assembly_kernel_name, num_vertices_per_cell,
      num_coordinates_per_vertex, num_dofs_per_cell0, num_dofs_per_cell1) + "\n"
    "\n"
    "extern \"C\" void __global__\n"
    "__launch_bounds__(" + std::to_string(max_threads_per_block) + ", " + std::to_string(min_blocks_per_multiprocessor) + ")\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    //"  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ nonzero_locations,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    //"    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation);\n"
    "\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        int64_t l = (((int64_t) (i / warpSize) *\n"
    "          " + std::to_string(num_dofs_per_cell0) + "ll + (int64_t) j) *\n"
    "          " + std::to_string(num_dofs_per_cell1) + "ll + (int64_t) k) *\n"
    "          warpSize + (i % warpSize);\n"
    "        int r = nonzero_locations[l];\n"
    "        if (r < 0) continue;\n"
    "        atomicAdd(&values[r],\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for computing a lookup table for the sparse matrix
/// non-zeros corresponding to the degrees of freedom of each mesh entity
std::string cuda_kernel_compute_lookup_table_rowwise(
  std::string assembly_kernel_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "compute_lookup_table_" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_dofs,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const int32_t* __restrict__ cells_per_dof_ptr,\n"
    "  const int32_t* __restrict__ cells_per_dof,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_rows,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  int64_t num_nonzero_locations,\n"
    "  int32_t* __restrict__ nonzero_locations,\n"
    "  int32_t* __restrict__ element_matrix_rows)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  assert(num_dofs == num_rows);\n"
    "\n"
    "  // Iterate over the global degrees of freedom of the test space\n"
    "  for (int i = thread_idx;\n"
    "    i < num_dofs;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    if (bc0 && bc0[i]) {\n"
    "      for (int p = cells_per_dof_ptr[i]; p < cells_per_dof_ptr[i+1]; p++) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (int64_t) ((p / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + k) * warpSize +\n"
    "            p % warpSize;\n"
    "          nonzero_locations[l] = -1;\n"
    "        }\n"
    "      }\n"
    "    } else {\n"
    "      // Iterate over the mesh cells containing the current degree of freedom\n"
    "      // TODO: What if the integral is only over part of the domain?\n"
    "      // How do we iterate over the \"active\" cells?\n"
    "      for (int p = cells_per_dof_ptr[i]; p < cells_per_dof_ptr[i+1]; p++) {\n"
    "        int32_t c = cells_per_dof[p];\n"
    "\n"
    "        const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "        const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "        // Find the row of the element matrix that contributes to\n"
    "        // the current global degree of freedom\n"
    "        int j = 0;\n"
    "        while (j < " + std::to_string(num_dofs_per_cell0) + " && i != dofs0[j])\n"
    "          j++;\n"
    "        assert(i == dofs0[j]);\n"
    "        element_matrix_rows[p] = j;\n"
    "\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (int64_t) ((p / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + k) * warpSize +\n"
    "            p % warpSize;\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) {\n"
    "            nonzero_locations[l] = -1;\n"
    "          } else {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[i+1] - row_ptr[i],\n"
    "              &column_indices[row_ptr[i]],\n"
    "              column, &r);\n"
    "            assert(!err);\n"
    "            nonzero_locations[l] = row_ptr[i] + r;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for rowwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_rowwise(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    cuda_kernel_compute_lookup_table_rowwise(
      assembly_kernel_name, num_vertices_per_cell,
      num_coordinates_per_vertex, num_dofs_per_cell0, num_dofs_per_cell1) + "\n"
    "\n"
    "extern \"C\" void __global__\n"
    "__launch_bounds__(" + std::to_string(max_threads_per_block) + ", " + std::to_string(min_blocks_per_multiprocessor) + ")\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    //"  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ cells_per_dof_ptr,\n"
    "  const int32_t* __restrict__ cells_per_dof,\n"
    "  const int32_t* __restrict__ nonzero_locations,\n"
    "  const int32_t* __restrict__ element_matrix_rows,\n"
    "  int32_t num_rows,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  // Iterate over the global degrees of freedom of the test space and\n"
    "  // the mesh cells containing them.\n"
    "  // TODO: What if the integral is only over part of the domain?\n"
    "  // How do we iterate over the \"active\" cells?\n"
    "  for (int p = thread_idx;\n"
    "       p < cells_per_dof_ptr[num_rows];\n"
    "       p += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = cells_per_dof[p];\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    //"    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation);\n"
    "\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      if (j != element_matrix_rows[p]) continue;\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        int64_t l = (int64_t) ((p / warpSize) *\n"
    "          " + std::to_string(num_dofs_per_cell1) + " + k) * warpSize +\n"
    "          p % warpSize;\n"
    "        int r = nonzero_locations[l];\n"
    "        if (r < 0) continue;\n"
    "        atomicAdd(&values[r],\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  enum assembly_kernel_type assembly_kernel_type)
{
  switch (assembly_kernel_type) {
  case ASSEMBLY_KERNEL_LOCAL:
    return cuda_kernel_assemble_matrix_cell_local(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case ASSEMBLY_KERNEL_GLOBAL:
    return cuda_kernel_assemble_matrix_cell_global(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case ASSEMBLY_KERNEL_LOOKUP_TABLE:
    return cuda_kernel_assemble_matrix_cell_lookup_table(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case ASSEMBLY_KERNEL_ROWWISE:
    return cuda_kernel_assemble_matrix_cell_rowwise(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  default:
    throw std::invalid_argument(
      "Invalid argument at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

/// CUDA C++ code for assembly of a matrix from an exterior facet integral

std::string cuda_kernel_assemble_matrix_exterior_facet(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  enum assembly_kernel_type assembly_kernel_type
)
{
  // For now we don't care about the assembly_kernel_type
  // We default to the global algorithm
  return 
    "extern \"C\" int printf(const char * format, ...);\n"
    "\n"
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_mesh_entities,\n"
    "  const int32_t* __restrict__ active_mesh_entities,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_local_rows,\n"
    "  int32_t num_local_columns,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  ufc_scalar_t* __restrict__ values,\n"
    "  const int32_t* __restrict__ offdiag_row_ptr,\n"
    "  const int32_t* __restrict__ offdiag_column_indices,\n"
    "  ufc_scalar_t* __restrict__ offdiag_values,\n"
    "  int32_t num_local_offdiag_columns,\n"
    "  const int32_t* __restrict__ colmap)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    " \n"
    "  for (int i = 2*thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += 2*blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_mesh_entities[i];\n"
    "    int32_t local_mesh_entity = active_mesh_entities[i+1];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      &local_mesh_entity,\n"
    "      quadrature_permutation);\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) continue;\n"
    "      if (row < num_local_rows) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) continue;\n"
    "          if (column < num_local_columns) {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[row+1] - row_ptr[row],\n"
    "              &column_indices[row_ptr[row]],\n"
    "              column, &r);\n"
    "            assert(!err && \"Failed to find column index in assemble_matrix_exterior_facet!\");\n"
    "            r += row_ptr[row];\n"
    "            atomicAdd(&values[r],\n"
    "              Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          } else {\n"
    "            /* Search for the correct column index in the column map\n"
    "             * of the off-diagonal part of the local matrix. */\n"
    "            int32_t colmap_idx = -1;\n"
    "            for (int q = 0; q < num_local_offdiag_columns; q++) {\n"
    "              if (column == colmap[q]) {\n"
    "                colmap_idx = q;\n"
    "                break;\n"
    "              }\n"
    "            }\n"
    "            assert(colmap_idx != -1);\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              offdiag_row_ptr[row+1] - offdiag_row_ptr[row],\n"
    "              &offdiag_column_indices[offdiag_row_ptr[row]],\n"
    "              colmap_idx, &r);\n"
    "            assert(!err && \"Failed to find offidag column index in assemble_matrix_exterior_facet!\");\n"
    "            r += offdiag_row_ptr[row];\n"
    "            atomicAdd(&offdiag_values[r],\n"
    "              Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n";  
}

std::string dump_arr(const std::string& name, const std::string& length, const std::string& fmt)
{
  return
    " printf(\""+name+": [\");\n"
    " for (int j=0; j <  "+length+"; j++) {\n"
    "   printf(\"%"+fmt+", \", "+name+"[j]);\n"
    "}\n"
    "printf(\"]\\n\");\n\n";
}

// Generate debugging code to print all assembly variables
std::string dump_assembly_vars(IntegralType integral_type)
{
  std::string body = "";
  std::string n_coords;
  switch (integral_type) {
    case IntegralType::interior_facet:
      n_coords = "2*num_vertices_per_cell*num_coordinates_per_vertex";
      body += dump_arr("coefficient_values_offsets", "num_coefficients+1", "d");
      body += dump_arr("cell_coeffs", "2*num_coeffs_per_cell", "f");
      body += dump_arr("cell_coeffs0", "num_coeffs_per_cell", "f");
      body += dump_arr("cell_coeffs1", "num_coeffs_per_cell", "f");
      break;
    case IntegralType::cell:
    case IntegralType::exterior_facet:
      n_coords = "num_vertices_per_cell*num_coordinates_per_vertex";
      body += dump_arr("coeff_cell", "num_coeffs_per_cell", "f");
      break;  
  }

  body +=
    dump_arr("cell_vertex_coordinates", n_coords, "f");

  return body;
}

// For cell and exterior facet integrals, packed coefficients
// are stored in contiguous blocks pertaining to a single cell
// However, interior facets require that values from each cell 
// are stored side by side e.g. (a1b1c1a2b2c2 -> a1a2b1b2c1c2)
// Instead of precomputing different arrays for each domain id,
// we simply do the interlacing on the fly to reduce the number of kernels
// and GPU memory usage
// returns a snippet that can be used inside a larger function
std::string interior_facet_pack_cell_coeffs(int32_t num_coeffs_per_cell)
{
  if (num_coeffs_per_cell <= 0)
    return "ufc_scalar_t* cell_coeffs = NULL;\n";
  return 
  "  ufc_scalar_t cell_coeffs[2*"+std::to_string(num_coeffs_per_cell)+"];\n"
  "  const ufc_scalar_t* cell_coeffs0 = &coeffs[c0*num_coeffs_per_cell];\n"
  "  const ufc_scalar_t* cell_coeffs1 = &coeffs[c1*num_coeffs_per_cell];\n"
  "  for (int j = 0; j < num_coefficients; j++) {\n"
  "    int offset = coefficient_values_offsets[j];\n"
  "    int coeff_size = coefficient_values_offsets[j+1]-offset;\n"
  "    for (int k = 0; k < coeff_size; k++) {\n"
  "      cell_coeffs[2*offset + k] = cell_coeffs0[offset+k];\n"
  "      cell_coeffs[2*offset + coeff_size + k] = cell_coeffs1[offset+k];\n"
  "    }\n"
  "  }\n";
}

// args needed for interior facet

std::string interior_facet_extra_args()
{
  return 
  "  int32_t num_mesh_entities_per_cell,\n"
  "  const uint8_t* __restrict__ facet_permutations,\n"
  "  int num_coefficients,\n"
  "  const int* __restrict__ coefficient_values_offsets,\n";
}

// body of tensor assembly for interior facet
std::string compute_interior_facet_tensor(
 std::string tabulate_tensor_function_name, 
 int32_t num_vertices_per_cell,
 int32_t num_coordinates_per_vertex,
 int32_t num_coeffs_per_cell,
 bool vector
)
{
  std::string body =
    interior_facet_pack_cell_coeffs(num_coeffs_per_cell) + 
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c0*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c1*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates["+std::to_string(num_coordinates_per_vertex*num_vertices_per_cell)+"+j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    uint8_t quadrature_permutation[2];\n"
    "    if (facet_permutations != NULL) {\n"
    "      quadrature_permutation[0] = facet_permutations[c0*num_mesh_entities_per_cell + facet0];\n"
    "      quadrature_permutation[1] = facet_permutations[c1*num_mesh_entities_per_cell + facet1];\n"
    "    }\n"
    "    else {\n"
    "      quadrature_permutation[0] = quadrature_permutation[1] = 0;\n"
    "    }\n" 
    "\n"
    "    int32_t local_mesh_entities[2] = {facet0, facet1};";
    if (vector) {
      return body +
      "    // Compute element matrix\n"
      "    " + tabulate_tensor_function_name + "(\n"
      "      xe,\n"
      "      cell_coeffs,\n"
      "      constant_values,\n"
      "      cell_vertex_coordinates,\n"
      "      local_mesh_entities,\n"
      "      quadrature_permutation);\n";        
    }
    return body +
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      cell_coeffs,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      local_mesh_entities,\n"
    "      quadrature_permutation);\n";

}

std::string get_interior_facet_joint_dofmaps(
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1
)
{
  return 
    "    int32_t dofs0[2*"+std::to_string(num_dofs_per_cell0)+"];\n"
    "    int32_t dofs1[2*"+std::to_string(num_dofs_per_cell1)+"];\n"
    "    {\n"
    "      const int32_t* dofs11 = &dofmap1[c1*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "      const int32_t* dofs00 = &dofmap0[c0*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "      const int32_t* dofs01 = &dofmap0[c1*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "      const int32_t* dofs10 = &dofmap1[c0*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "      for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "        dofs0[j] = dofs00[j];\n"
    "        dofs0[j+"+std::to_string(num_dofs_per_cell0)+"] = dofs01[j];\n"
    "      }\n"
    "      for (int j = 0; j < " + std::to_string(num_dofs_per_cell1) + "; j++) {\n"
    "        dofs1[j] = dofs10[j];\n"
    "        dofs1[j+"+std::to_string(num_dofs_per_cell1)+"] = dofs11[j];\n"
    "      }\n"
    "    }\n";
}

std::string get_interior_facet_joint_dofmap(
  int32_t num_dofs_per_cell
)
{
  return
    "    int32_t dofs[2*"+std::to_string(num_dofs_per_cell)+"];\n"
    "    {\n"
    "      const int32_t* dofs0 = &dofmap[c0*" + std::to_string(num_dofs_per_cell) + "];\n"
    "      const int32_t* dofs1 = &dofmap[c1*" + std::to_string(num_dofs_per_cell) + "];\n"
    "      for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "        dofs[j] = dofs0[j];\n"
    "        dofs[j+"+std::to_string(num_dofs_per_cell)+"] = dofs1[j];\n"
    "      }\n"
    "    }\n";
}

std::string cuda_kernel_assemble_matrix_interior_facet(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  int32_t num_coeffs_per_cell,
  enum assembly_kernel_type assembly_kernel_type
)
{

  return ""
    "extern \"C\" int printf(const char * format, ...);\n"
    "\n"
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_mesh_entities,\n"
    "  const int32_t* __restrict__ active_mesh_entities,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    + interior_facet_extra_args() +
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_local_rows,\n"
    "  int32_t num_local_columns,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  ufc_scalar_t* __restrict__ values,\n"
    "  const int32_t* __restrict__ offdiag_row_ptr,\n"
    "  const int32_t* __restrict__ offdiag_column_indices,\n"
    "  ufc_scalar_t* __restrict__ offdiag_values,\n"
    "  int32_t num_local_offdiag_columns,\n"
    "  const int32_t* __restrict__ colmap)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  assert(num_coeffs_per_cell == " + std::to_string(num_coeffs_per_cell) + ");\n"
    "  double cell_vertex_coordinates[2*" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[4*" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    " \n"
    "  for (int i = 4*thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += 4*blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c0 = active_mesh_entities[i];\n"
    "    int32_t c1 = active_mesh_entities[i+2];\n"
    "    int32_t facet0 = active_mesh_entities[i+1];\n"
    "    int32_t facet1 = active_mesh_entities[i+3];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < 2*" + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < 2*" + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[2*j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    + compute_interior_facet_tensor(
       tabulate_tensor_function_name, 
       num_vertices_per_cell,
       num_coordinates_per_vertex,
       num_coeffs_per_cell
    )
    + get_interior_facet_joint_dofmaps(num_dofs_per_cell0, num_dofs_per_cell1) +
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "\n"
    "    for (int j = 0; j < 2*" + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) continue;\n"
    "      if (row < num_local_rows) {\n"
    "        for (int k = 0; k < 2*" + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) continue;\n"
    "          if (column < num_local_columns) {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[row+1] - row_ptr[row],\n"
    "              &column_indices[row_ptr[row]],\n"
    "              column, &r);\n"
    "            assert(!err && \"Failed to find column index in assemble_matrix_interior_facet!\");\n"
    "            r += row_ptr[row];\n"
    "            atomicAdd(&values[r],\n"
    "              Ae[j*2*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          } else {\n"
    "            /* Search for the correct column index in the column map\n"
    "             * of the off-diagonal part of the local matrix. */\n"
    "            int32_t colmap_idx = -1;\n"
    "            for (int q = 0; q < num_local_offdiag_columns; q++) {\n"
    "              if (column == colmap[q]) {\n"
    "                colmap_idx = q;\n"
    "                break;\n"
    "              }\n"
    "            }\n"
    "            assert(colmap_idx != -1);\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              offdiag_row_ptr[row+1] - offdiag_row_ptr[row],\n"
    "              &offdiag_column_indices[offdiag_row_ptr[row]],\n"
    "              colmap_idx, &r);\n"
    "            assert(!err && \"Failed to find offdiag column index in assemble_matrix_interior_facet!\");\n"
    "            r += offdiag_row_ptr[row];\n"
    "            atomicAdd(&offdiag_values[r],\n"
    "              Ae[j*2*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n";  
}

/// CUDA C++ code for assembly of a matrix from a form integral
std::string cuda_kernel_assemble_matrix(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  IntegralType integral_type,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  int32_t num_coeffs_per_cell,
  enum assembly_kernel_type assembly_kernel_type)
{
  switch (integral_type) {
  case IntegralType::cell:
    return cuda_kernel_assemble_matrix_cell(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      assembly_kernel_type);
  case IntegralType::exterior_facet:
    return cuda_kernel_assemble_matrix_exterior_facet(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      assembly_kernel_type);
  case IntegralType::interior_facet:
    return cuda_kernel_assemble_matrix_interior_facet(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      num_coeffs_per_cell,
      assembly_kernel_type);
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

static const char * nvrtc_options_gpuarch(
    CUjit_target target)
{
    switch (target) {
    case CU_TARGET_COMPUTE_30: return "--gpu-architecture=compute_30";
    case CU_TARGET_COMPUTE_32: return "--gpu-architecture=compute_32";
    case CU_TARGET_COMPUTE_35: return "--gpu-architecture=compute_35";
    case CU_TARGET_COMPUTE_37: return "--gpu-architecture=compute_37";
    case CU_TARGET_COMPUTE_50: return "--gpu-architecture=compute_50";
    case CU_TARGET_COMPUTE_52: return "--gpu-architecture=compute_52";
    case CU_TARGET_COMPUTE_53: return "--gpu-architecture=compute_53";
    case CU_TARGET_COMPUTE_60: return "--gpu-architecture=compute_60";
    case CU_TARGET_COMPUTE_61: return "--gpu-architecture=compute_61";
    case CU_TARGET_COMPUTE_62: return "--gpu-architecture=compute_62";
    case CU_TARGET_COMPUTE_70: return "--gpu-architecture=compute_70";
    case CU_TARGET_COMPUTE_72: return "--gpu-architecture=compute_72";
    case CU_TARGET_COMPUTE_75: return "--gpu-architecture=compute_75";
    case CU_TARGET_COMPUTE_80: return "--gpu-architecture=compute_80";
    case CU_TARGET_COMPUTE_86: return "--gpu-architecture=compute_86";
    case CU_TARGET_COMPUTE_87: return "--gpu-architecture=compute_87";
    case CU_TARGET_COMPUTE_89: return "--gpu-architecture=compute_89";
    case CU_TARGET_COMPUTE_90: return "--gpu-architecture=compute_90";
    default: return "";
    }
}

/// Configure compiler options for CUDA C++ code
static const char** nvrtc_compiler_options(
  int* out_num_compile_options,
  CUjit_target target,
  bool debug)
{
  int num_compile_options;
  static const char* default_compile_options[] = {
    "--device-as-default-execution-space",
    nvrtc_options_gpuarch(target)};
  static const char* debug_compile_options[] = {
    "--device-as-default-execution-space",
    nvrtc_options_gpuarch(target),
    "--device-debug",
    "--generate-line-info"};

  const char** compile_options;
  if (debug) {
    compile_options = debug_compile_options;
    num_compile_options =
      sizeof(debug_compile_options) /
      sizeof(*debug_compile_options);
  } else {
    compile_options = default_compile_options;
    num_compile_options =
      sizeof(default_compile_options) /
      sizeof(*default_compile_options);
  }

  *out_num_compile_options = num_compile_options;
  return compile_options;
}

} // namespace

//-----------------------------------------------------------------------------
CUresult dolfinx::fem::launch_cuda_kernel(
  CUfunction kernel,
  unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
  unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
  unsigned int shared_mem_size_per_thread_block,
  CUstream stream,
  void** kernel_parameters,
  void** extra,
  bool verbose)
{
  if (verbose) {
    int max_threads_per_block;
    cuFuncGetAttribute(
      &max_threads_per_block,
      CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      kernel);
    int shared_size_bytes;
    cuFuncGetAttribute(
      &shared_size_bytes,
      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
      kernel);
    int max_dynamic_shared_size_bytes;
    cuFuncGetAttribute(
      &max_dynamic_shared_size_bytes,
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      kernel);
    int const_size_bytes;
    cuFuncGetAttribute(
      &const_size_bytes,
      CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
      kernel);
    int local_size_bytes;
    cuFuncGetAttribute(
      &local_size_bytes,
      CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
      kernel);
    int num_regs;
    cuFuncGetAttribute(
      &num_regs,
      CU_FUNC_ATTRIBUTE_NUM_REGS,
      kernel);
    int ptx_version;
    cuFuncGetAttribute(
      &ptx_version,
      CU_FUNC_ATTRIBUTE_PTX_VERSION,
      kernel);
    int binary_version;
    cuFuncGetAttribute(
      &binary_version,
      CU_FUNC_ATTRIBUTE_BINARY_VERSION,
      kernel);
    int cache_mode_ca;
    cuFuncGetAttribute(
      &cache_mode_ca,
      CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
      kernel);
    int preferred_shared_memory_carveout;
    cuFuncGetAttribute(
      &preferred_shared_memory_carveout,
      CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
      kernel);

    fprintf(stderr, "Launching kernel with "
            "%dx%dx%d grid of thread blocks (%d thread blocks), "
            "%dx%dx%d threads in each block (max %d threads per block), "
            "%d bytes of statically allocated shared memory per block, "
            "maximum of %d bytes of dynamic shared memory per block, "
            "%d bytes of constant memory, "
            "%d bytes of local memory per thread, "
            "%d registers per thread, "
            "PTX version %d, binary version %d, "
            "%s, preferred shared memory carve-out: %d.\n",
            grid_dim_x, grid_dim_y, grid_dim_z, grid_dim_x*grid_dim_y*grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z, max_threads_per_block,
            shared_size_bytes, max_dynamic_shared_size_bytes,
            const_size_bytes, local_size_bytes, num_regs,
            ptx_version, binary_version,
            cache_mode_ca ? "global loads cached in L1 cache" : "global loads not cached in L1 cache",
            preferred_shared_memory_carveout);
  }

  return cuLaunchKernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, extra);
}
//-----------------------------------------------------------------------------


/// Compile assembly kernel for a form integral
CUDA::Module dolfinx::fem::compile_form_integral_kernel(
  const CUDA::Context& cuda_context,
  CUjit_target target,
  int form_rank,
  IntegralType integral_type,
  std::function<void(int*, const char***, const char***,
                     const char**, const char**)>
  cuda_tabulate,
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
  std::string& factory_name)
{
  // Obtain the automatically generated CUDA C++ code for the
  // element matrix kernel (tabulate_tensor).
  int num_program_headers;
  const char** program_headers;
  const char** program_include_names;
  const char* tabulate_tensor_src;
  const char* tabulate_tensor_function_name;
  cuda_tabulate(
    &num_program_headers, &program_headers,
    &program_include_names, &tabulate_tensor_src,
    &tabulate_tensor_function_name);
  // Generate CUDA C++ code for the assembly kernel

  // extract the factory/integral name from the tabulate tensor name
  factory_name = std::string(tabulate_tensor_function_name);
  std::string pref = std::string("tabulate_tensor");
  if ((factory_name.find(pref) == 0) && (factory_name.length() > pref.length())) {
    factory_name = factory_name.replace(0, pref.length(), std::string(""));
  }

  std::string assembly_kernel_name =
    std::string("assemble_") + std::string(factory_name);
  std::string lift_bc_kernel_name =
    std::string("lift_bc_") + std::string(factory_name);

  switch (integral_type) {
    case IntegralType::interior_facet:
      assembly_kernel_name += std::string("_if");
      break;
    case IntegralType::exterior_facet:
      assembly_kernel_name += std::string("_ef");
      break;
    case IntegralType::cell:
      assembly_kernel_name += std::string("_c");
      break;
  }

  std::string assembly_kernel_src =
    std::string(tabulate_tensor_src) + "\n"
    "typedef int int32_t;\n"
    "typedef long long int int64_t;\n"
    "\n";

  switch (form_rank) {
  case 1:
    assembly_kernel_name += "_vec";
    assembly_kernel_src += cuda_kernel_assemble_vector(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      integral_type,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_coeffs_per_cell
      ) + "\n";
    break;
  case 2:
    assembly_kernel_name += "_mat";
    assembly_kernel_src += cuda_kernel_binary_search() + "\n"
      "\n";
    assembly_kernel_src += cuda_kernel_assemble_matrix(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      integral_type,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      num_coeffs_per_cell,
      assembly_kernel_type) + "\n"
      "\n";
    assembly_kernel_src += cuda_kernel_lift_bc(
      lift_bc_kernel_name,
      tabulate_tensor_function_name,
      integral_type,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      num_coeffs_per_cell
      ) + "\n";
    break;
  default:
    throw std::runtime_error(
      "Forms of rank " + std::to_string(form_rank) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  // Configure compiler options
  int num_compile_options;
  const char** compile_options =
    nvrtc_compiler_options(&num_compile_options, target, debug);

  // Compile CUDA C++ code to PTX assembly
  const char* program_name = factory_name.c_str();
  std::string ptx = CUDA::compile_cuda_cpp_to_ptx(
    program_name, num_program_headers, program_headers,
    program_include_names, num_compile_options, compile_options,
    assembly_kernel_src.c_str(), cudasrcdir, verbose);

  // Load the PTX assembly as a module
  int num_module_load_options = 0;
  CUjit_option * module_load_options = NULL;
  void ** module_load_option_values = NULL;
  return CUDA::Module(
    cuda_context, ptx, target,
    num_module_load_options,
    module_load_options,
    module_load_option_values,
    verbose,
    debug);
}


std::string dolfinx::fem::to_string(IntegralType integral_type)
{
  switch (integral_type) {
  case IntegralType::cell: return "cell";
  case IntegralType::exterior_facet: return "exterior_facet";
  case IntegralType::interior_facet: return "interior_facet";
  default: return "unknown";
  }
}

//-----------------------------------------------------------------------------
std::string dolfinx::fem::cuda_kernel_binary_search(void)
{
  return
    "/**\n"
    " * `binary_search()` performs a binary search to find the location\n"
    " * of a given element in a sorted array of integers.\n"
    " */\n"
    "extern \"C\" __device__ int binary_search(\n"
    "  int num_elements,\n"
    "  const int * __restrict__ elements,\n"
    "  int key,\n"
    "  int * __restrict__ out_index)\n"
    "{\n"
    "  if (num_elements <= 0)\n"
    "    return -1;\n"
    "\n"
    "  int p = 0;\n"
    "  int q = num_elements;\n"
    "  int r = (p + q) / 2;\n"
    "  while (p < q) {\n"
    "    if (elements[r] == key) break;\n"
    "    else if (elements[r] < key) p = r + 1;\n"
    "    else q = r - 1;\n"
    "    r = (p + q) / 2;\n"
    "  }\n"
    "  if (elements[r] != key)\n"
    "    return -1;\n"
    "  *out_index = r;\n"
    "  return 0;\n"
    "}\n";
}
//-----------------------------------------------------------------------------
cuda_kern dolfinx::fem::get_cuda_wrapper(std::array<std::map<int, cuda_kern>, 4>& cuda_wrappers,
	       	IntegralType type, int i)
{
  auto integrals = cuda_wrappers[static_cast<std::size_t>(type)];
  if (auto it = integrals.find(i); it != integrals.end())
    return it->second;
  else
    throw std::runtime_error("No kernel for requested domain index.");
}

//-----------------------------------------------------------------------------
