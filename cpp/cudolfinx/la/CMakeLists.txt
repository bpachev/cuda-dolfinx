set(HEADERS_la
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMatrix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDASeqMatrix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAVector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/petsc.h
    PARENT_SCOPE
)

target_sources(
  cudolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CUDAMatrix.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/CUDASeqMatrix.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/CUDAVector.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/petsc.cpp
)
