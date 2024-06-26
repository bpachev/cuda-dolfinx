set(HEADERS_mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMeshEntities.h
    PARENT_SCOPE
)

target_sources(
  cudolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMesh.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMeshEntities.cpp
)
