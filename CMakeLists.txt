cmake_minimum_required(VERSION 3.11)
project(hello)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules" CACHE STRING "")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)


find_package(VexCL)



option(VECTORIZE_COMPUTATIONS "Explicit vectorization of the computation (default ON)" ON)
if(VECTORIZE_COMPUTATIONS)
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-DVECTORIZE_COMPUTATIONS>)
	option(EIGEN_VECTORIZATION "Use Eigen as the vectorization engine (default ON)" ON)
	if (EIGEN_VECTORIZATION)
		add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-DEIGEN_VECTORIZATION>)
	endif(EIGEN_VECTORIZATION)
endif(VECTORIZE_COMPUTATIONS)



# Eigen
find_package(Eigen3 3.3.3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")

add_executable(main main.cpp)
target_link_libraries(main "C:/Utils/boost_1_69_0/stage/lib/Debug/*")

target_link_libraries(main VexCL::CUDA)

