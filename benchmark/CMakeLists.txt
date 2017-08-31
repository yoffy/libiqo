cmake_minimum_required(VERSION 3.5)
project(benchmark)

include(CheckIncludeFileCXX)

add_executable(benchmark benchmark.cpp)


# check include files
set(CMAKE_REQUIRED_FLAGS "-lopencv_core")
CHECK_INCLUDE_FILE_CXX(opencv2/opencv.hpp HAVE_OPENCV_HPP)
set(CMAKE_REQUIRED_FLAGS "")
CHECK_INCLUDE_FILE_CXX(ipp.h HAVE_IPP_H)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)


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
if ( HAVE_OPENCV_HPP )
    target_link_libraries(benchmark LINK_PUBLIC opencv_core opencv_imgproc)
endif()
if ( HAVE_IPP_H )
    target_link_libraries(benchmark LINK_PUBLIC ippi ipps ippcore)
endif()
