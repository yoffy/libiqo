cmake_minimum_required(VERSION 3.5)
project(iqo)

include(CheckCXXCompilerFlag)
include(CheckFunctionExists)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib/)


# default build type
if ( NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release)
endif()


# options
option(WITH_OPENMP "Use OpenMP" OFF)


# CXX base flags
if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -fno-exceptions -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wno-sign-conversion")
elseif ( MSVC )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /Zi")
endif()


# Release options
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-Ofast" COMPILER_SUPPORTS_OFAST)
    CHECK_CXX_COMPILER_FLAG("-O3"    COMPILER_SUPPORTS_O3)
    if ( COMPILER_SUPPORTS_OFAST )
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
    elseif ( COMPILER_SUPPORTS_O3 )
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    endif()
elseif ( MSVC )
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox /fp:fast /GS- /MT")
endif()


# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
elseif( MSVC )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /MT")
endif()


# OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_BASE_FLAGS} ${CMAKE_CXX_OPT_FLAGS}")
if ( WITH_OPENMP )
    find_package(OpenMP)
    if ( OPENMP_FOUND )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


# include and sub projects
include_directories(include)
add_subdirectory(src)
add_subdirectory(sample)
