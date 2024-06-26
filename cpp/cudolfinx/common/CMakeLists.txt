set(HEADERS_common
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDA.h
    PARENT_SCOPE
)

target_sources(
  cudolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CUDA.cpp
)
