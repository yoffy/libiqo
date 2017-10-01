include(CheckCXXSymbolExists)

add_library(iqo IQOHWCap.cpp
                IQOAreaResizer.cpp
                IQOAreaResizerImpl_Generic.cpp
                IQOAreaResizerImpl_SSE4_1.cpp
                IQOAreaResizerImpl_AVX2FMA.cpp
                IQOAreaResizerImpl_AVX512.cpp
                IQOAreaResizerImpl_NEON.cpp
                IQOLinearResizer.cpp
                IQOLinearResizerImpl_Generic.cpp
                IQOLinearResizerImpl_SSE4_1.cpp
                IQOLinearResizerImpl_AVX2FMA.cpp
                IQOLinearResizerImpl_AVX512.cpp
                IQOLinearResizerImpl_NEON.cpp
                IQOLanczosResizer.cpp
                IQOLanczosResizerImpl_Generic.cpp
                IQOLanczosResizerImpl_SSE4_1.cpp
                IQOLanczosResizerImpl_AVX2FMA.cpp
                IQOLanczosResizerImpl_AVX512.cpp
                IQOLanczosResizerImpl_NEON.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)


# CXX base flags

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -fno-exceptions")
endif()


# Intel

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=core2 -msse4.1" COMPILER_SUPPORTS_MARCH_CORE2_MSSE4_1)
    CHECK_CXX_COMPILER_FLAG("-mtune=westmere" COMPILER_SUPPORTS_MTUNE_WESTMERE)
    if ( COMPILER_SUPPORTS_MTUNE_WESTMERE )
        set(MTUNE_SSE4_1 "-mtune=westmere")
    endif()
    if ( COMPILER_SUPPORTS_MARCH_CORE2_MSSE4_1 )
        set_property(SOURCE IQOAreaResizerImpl_SSE4_1.cpp     APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1 ${MTUNE_SSE4_1}")
        set_property(SOURCE IQOLinearResizerImpl_SSE4_1.cpp   APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1 ${MTUNE_SSE4_1}")
        set_property(SOURCE IQOLanczosResizerImpl_SSE4_1.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1 ${MTUNE_SSE4_1}")
    endif()
endif()

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=core-avx2" COMPILER_SUPPORTS_MARCH_CORE_AVX2)
    CHECK_CXX_COMPILER_FLAG("-mtune=skylake"   COMPILER_SUPPORTS_MTUNE_SKYLAKE)
    if ( COMPILER_SUPPORTS_MTUNE_SKYLAKE )
        set(MTUNE_AVX2 "-mtune=skylake")
    endif()
    if ( COMPILER_SUPPORTS_MARCH_CORE_AVX2 )
        set_property(SOURCE IQOAreaResizerImpl_AVX2FMA.cpp      APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2 ${MTUNE_AVX2}")
        set_property(SOURCE IQOLinearResizerImpl_AVX2FMA.cpp    APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2 ${MTUNE_AVX2}")
        set_property(SOURCE IQOLanczosResizerImpl_AVX2FMA.cpp   APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2 ${MTUNE_AVX2}")
    endif()
elseif ( MSVC )
    CHECK_CXX_COMPILER_FLAG("/arch:AVX2"       COMPILER_SUPPORTS_ARCH_AVX2)
    if ( COMPILER_SUPPORTS_ARCH_AVX2 )
        set_property(SOURCE IQOAreaResizerImpl_AVX2FMA.cpp      APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX2")
        set_property(SOURCE IQOLinearResizerImpl_AVX2FMA.cpp    APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX2")
        set_property(SOURCE IQOLanczosResizerImpl_AVX2FMA.cpp   APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX2")
    endif()
endif()

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=skylake-avx512" COMPILER_SUPPORTS_MARCH_SKYLAKE_AVX512)
    CHECK_CXX_COMPILER_FLAG("-mtune=skylake-avx512" COMPILER_SUPPORTS_MTUNE_SKYLAKE_AVX512)
    if ( COMPILER_SUPPORTS_MTUNE_SKYLAKE_AVX512 )
        set(MTUNE_AVX512 "-mtune=skylake-avx512")
    endif()
    if ( COMPILER_SUPPORTS_MARCH_SKYLAKE_AVX512 )
        set_property(SOURCE IQOAreaResizerImpl_AVX512.cpp       APPEND_STRING PROPERTY COMPILE_FLAGS "-march=skylake-avx512 ${MTUNE_AVX512}")
        set_property(SOURCE IQOLinearResizerImpl_AVX512.cpp     APPEND_STRING PROPERTY COMPILE_FLAGS "-march=skylake-avx512 ${MTUNE_AVX512}")
        set_property(SOURCE IQOLanczosResizerImpl_AVX512.cpp    APPEND_STRING PROPERTY COMPILE_FLAGS "-march=skylake-avx512 ${MTUNE_AVX512}")
    endif()
elseif ( MSVC )
    # check to support Skylake-X (not KNL)
    set(CMAKE_REQUIRED_FLAGS    "/arch:AVX512")
    set(CMAKE_REQUIRED_INCLUDES "immintrin.h")
    # workaround: It depend to be defined macro "_mm512_cvt_roundps_epi32" in zmmintrin.h,
    #             because check_function_exists and check_cxx_symbol_exists can not determine builtin function.
    check_cxx_symbol_exists(_mm512_cvt_roundps_epi32 "immintrin.h" COMPILER_HAVE_MM512_CVT_ROUNDPS_EPI32)
    if ( COMPILER_HAVE_MM512_CVT_ROUNDPS_EPI32 )
        CHECK_CXX_COMPILER_FLAG("/arch:AVX512"          COMPILER_SUPPORTS_ARCH_AVX512)
        if ( COMPILER_SUPPORTS_ARCH_AVX512 )
            set_property(SOURCE IQOAreaResizerImpl_AVX512.cpp       APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX512")
            set_property(SOURCE IQOLinearResizerImpl_AVX512.cpp     APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX512")
            set_property(SOURCE IQOLanczosResizerImpl_AVX512.cpp    APPEND_STRING PROPERTY COMPILE_FLAGS "/arch:AVX512")
        endif()
    endif()
endif()


# ARM

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    CHECK_CXX_COMPILER_FLAG("-march=armv7-a -mfpu=neon" COMPILER_SUPPORTS_MARCH_ARMV7_A_NEON)
    if ( COMPILER_SUPPORTS_MARCH_ARMV7_A_NEON )
        set_property(SOURCE IQOAreaResizerImpl_NEON.cpp     APPEND_STRING PROPERTY COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
        set_property(SOURCE IQOLinearResizerImpl_NEON.cpp   APPEND_STRING PROPERTY COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
        set_property(SOURCE IQOLanczosResizerImpl_NEON.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
    endif()
endif()
