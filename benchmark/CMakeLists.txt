# benchmak
#
# Optional Variables:
#     OPENCV_DIR  Path to OpenCV. It is parent directory of include directory.
#     IPPROOT     Path to Intel IPP. Set ippvars.sh (ippvars.bat in Windows)

cmake_minimum_required(VERSION 3.5)
project(benchmark)

include(CheckIncludeFileCXX)

add_executable(benchmark benchmark.cpp)


# default build type
if ( NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release)
endif()


# CXX base flags
if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-exceptions -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wno-sign-conversion")
elseif ( MSVC )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /Zi")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _CRT_SECURE_NO_WARNINGS=1")
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


# linker options
if ( MSVC )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG")
endif()
target_link_libraries(benchmark LINK_PUBLIC iqo)


# OpenCV
find_package(OpenCV COMPONENTS core imgproc)
if ( OPENCV_FOUND )
    set(HAVE_OPENCV_HPP ON)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(benchmark LINK_PUBLIC ${OpenCV_LIBS})
endif()


# Intel IPP
find_path(IPP_H_PATH NAMES ipp.h PATHS $ENV{IPPROOT}/include NO_DEFAULT_PATH NO_CMAKE_PATH)
if ( IPP_H_PATH )
    set(HAVE_IPP_H ON)
    set(IPP_INCLUDE_DIR $ENV{IPPROOT}/include)
    set(IPP_LIB_DIR $ENV{IPPROOT}/lib/$ENV{IPP_TARGET_ARCH})
    message(STATUS "IPP_INCLUDE: ${IPP_INCLUDE_DIR}")
    message(STATUS "IPP_LIB: ${IPP_LIB_DIR}")
    include_directories(${IPP_INCLUDE_DIR})
    link_directories(${IPP_LIB_DIR})
    find_library(IPP_IPPI ippi)
    find_library(IPP_IPPS ipps)
    find_library(IPP_IPPCORE ippcore)
    target_link_libraries(benchmark LINK_PUBLIC ${IPP_IPPI} ${IPP_IPPS} ${IPP_IPPCORE})
endif()


# generate config.h
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)
