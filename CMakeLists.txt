cmake_minimum_required(VERSION 3.6)
project(iqo)

include(CheckCXXCompilerFlag)

# CMAKE_CXX_BASE_FLAGS
if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    set(CMAKE_CXX_BASE_FLAGS "${CMAKE_CXX_BASE_FLAGS} -std=c++98 -Wall -Wextra -fno-exceptions -g")
endif()

# CMAKE_CXX_OPT_FLAGS
set(CMAKE_OPT_FLAGS "${CMAKE_OPT_FLAGS} -D NDEBUG")
CHECK_CXX_COMPILER_FLAG("-Ofast" COMPILER_SUPPORTS_OFAST)
CHECK_CXX_COMPILER_FLAG("-O3"    COMPILER_SUPPORTS_O3)
if ( COMPILER_SUPPORTS_OFAST )
    set(CMAKE_CXX_OPT_FLAGS "${CMAKE_CXX_OPT_FLAGS} -Ofast")
elseif ( COMPILER_SUPPORTS_O3 )
    set(CMAKE_CXX_OPT_FLAGS "${CMAKE_CXX_OPT_FLAGS} -O3")
endif()

# CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_BASE_FLAGS} ${CMAKE_CXX_OPT_FLAGS}")

# include and sub projects
include_directories(include)
add_subdirectory(src)
add_subdirectory(sample)
