cmake_minimum_required(VERSION 3.5)
project(iqo)

include(CheckCXXCompilerFlag)

# options
option(WITH_OPENMP "Use OpenMP" OFF)


# CMAKE_CXX_BASE_FLAGS
if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    add_compile_options(-std=c++98 -Wall -Wextra -fno-exceptions -g)
endif()


# CMAKE_CXX_OPT_FLAGS
set(CMAKE_OPT_FLAGS "${CMAKE_OPT_FLAGS} -D NDEBUG")
CHECK_CXX_COMPILER_FLAG("-Ofast" COMPILER_SUPPORTS_OFAST)
CHECK_CXX_COMPILER_FLAG("-O3"    COMPILER_SUPPORTS_O3)
if ( COMPILER_SUPPORTS_OFAST )
    add_compile_options(-Ofast)
elseif ( COMPILER_SUPPORTS_O3 )
    add_compile_options(-O3)
endif()


# CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_BASE_FLAGS} ${CMAKE_CXX_OPT_FLAGS}")
find_package(OpenMP)
if ( WITH_OPENMP AND OPENMP_FOUND )
    add_compile_options(${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# include and sub projects
include_directories(include)
add_subdirectory(src)
add_subdirectory(sample)
