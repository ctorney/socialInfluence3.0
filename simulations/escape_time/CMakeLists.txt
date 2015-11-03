
# Top-Level CmakeLists.txt
cmake_minimum_required(VERSION 2.6)
PROJECT( INF )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} . )


find_package(VTK QUIET)
find_package(CUDA REQUIRED)
find_package(GSL REQUIRED)
include(${VTK_USE_FILE})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch sm_20 )

ADD_SUBDIRECTORY( visuals )

INCLUDE_DIRECTORIES(${INF_SOURCE_DIR}/include)

CUDA_ADD_EXECUTABLE(socialInfluence main switcherKernel.cu cnpy.cpp)

TARGET_LINK_LIBRARIES(socialInfluence plotGrid ${VTK_LIBRARIES} ${GSL_LIBRARIES})
