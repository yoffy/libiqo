add_library(iqo IQOHWCap.cpp
                IQOLinearResizer.cpp
                IQOLinearResizerImpl_Generic.cpp
                IQOAreaResizer.cpp
                IQOAreaResizerImpl_Generic.cpp
                IQOAreaResizerImpl_SSE4_1.cpp
                IQOAreaResizerImpl_AVX2FMA.cpp
                IQOAreaResizerImpl_AVX512.cpp
                IQOAreaResizerImpl_NEON.cpp
                IQOLanczosResizer.cpp
                IQOLanczosResizerImpl_Generic.cpp
                IQOLanczosResizerImpl_SSE4_1.cpp
                IQOLanczosResizerImpl_AVX2FMA.cpp
                IQOLanczosResizerImpl_AVX512.cpp
                IQOLanczosResizerImpl_NEON.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)


# Intel

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=core2 -msse4.1" COMPILER_SUPPORTS_MARCH_CORE2_MSSE4_1)
    if ( COMPILER_SUPPORTS_MARCH_CORE2_MSSE4_1 )
        set_property(SOURCE IQOAreaResizerImpl_SSE4_1.cpp     APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1")
        set_property(SOURCE IQOLanczosResizerImpl_SSE4_1.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1")
    endif()
endif()

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=core-avx2" COMPILER_SUPPORTS_MARCH_CORE_AVX2)
    if ( COMPILER_SUPPORTS_MARCH_CORE_AVX2 )
        set_property(SOURCE IQOAreaResizerImpl_AVX2FMA.cpp      APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2")
        set_property(SOURCE IQOLanczosResizerImpl_AVX2FMA.cpp   APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2")
    endif()
elseif ( MSVC )
    CHECK_CXX_COMPILER_FLAG("/arch:AVX2"       COMPILER_SUPPORTS_ARCH_AVX2)
    if ( COMPILER_SUPPORTS_ARCH_AVX2 )
        set_property(SOURCE IQOAreaResizerImpl_AVX2FMA.cpp      APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX2")
        set_property(SOURCE IQOLanczosResizerImpl_AVX2FMA.cpp   APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX2")
    endif()
endif()

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=skylake-avx512" COMPILER_SUPPORTS_MARCH_SKYLAKE_AVX512)
    if ( COMPILER_SUPPORTS_MARCH_SKYLAKE_AVX512 )
        set_property(SOURCE IQOAreaResizerImpl_AVX512.cpp       APPEND_STRING PROPERTY COMPILE_FLAGS "-march=skylake-avx512")
        set_property(SOURCE IQOLanczosResizerImpl_AVX512.cpp    APPEND_STRING PROPERTY COMPILE_FLAGS "-march=skylake-avx512")
    endif()
elseif ( MSVC )
    # check to support Skylake-X (not KNL)
    set(CMAKE_REQUIRED_FLAGS    "/arch:AVX512")
    set(CMAKE_REQUIRED_INCLUDES "immintrin.h")
    check_function_exists(_mm512_cvt_roundps_epi32 COMPILER_HAVE_MM512_CVT_ROUNDPS_EPI32)
    if ( COMPILER_HAVE_MM512_CVT_ROUNDPS_EPI32 )
        CHECK_CXX_COMPILER_FLAG("/arch:AVX512"          COMPILER_SUPPORTS_ARCH_AVX512)
        if ( COMPILER_SUPPORTS_ARCH_AVX512 )
            set_property(SOURCE IQOAreaResizerImpl_AVX512.cpp       APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX512")
            set_property(SOURCE IQOLanczosResizerImpl_AVX512.cpp    APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX512")
        endif()
    endif()
endif()


# ARM

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=armv7-a -mfpu=neon" COMPILER_SUPPORTS_MARCH_ARMV7_A_NEON)
    if ( COMPILER_SUPPORTS_MARCH_ARMV7_A_NEON )
        set_property(SOURCE IQOAreaResizerImpl_NEON.cpp     APPEND_STRING PROPERTY COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
        set_property(SOURCE IQOLanczosResizerImpl_NEON.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
    endif()
endif()