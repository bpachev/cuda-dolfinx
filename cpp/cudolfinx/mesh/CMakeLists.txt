set(HEADERS_mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMeshEntities.h
    ${CMAKE_CURRENT_SOURCE_DIR}/util.h
    PARENT_SCOPE
)

target_sources(
   cudolfinx
   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
)

