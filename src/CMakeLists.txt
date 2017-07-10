add_library(iqo IQOHWCap.cpp
                IQOLanczosResizer.cpp
                IQOLanczosResizerImpl_Generic.cpp
                IQOLanczosResizerImpl_SSE4_1.cpp
                IQOLanczosResizerImpl_AVX2FMA.cpp
				IQOLanczosResizerImpl_AVX512.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

CHECK_CXX_COMPILER_FLAG("-march=core2 -msse4.1" COMPILER_SUPPORTS_MARCH_CORE2_MSSE4_1)
if ( COMPILER_SUPPORTS_MARCH_CORE2_MSSE4_1 )
	set_property(SOURCE IQOLanczosResizerImpl_SSE4_1.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1")
endif()

CHECK_CXX_COMPILER_FLAG("-march=core-avx2" COMPILER_SUPPORTS_MARCH_CORE_AVX2)
if ( COMPILER_SUPPORTS_MARCH_CORE_AVX2 )
	set_property(SOURCE IQOLanczosResizerImpl_AVX2FMA.cpp APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2")
endif()

CHECK_CXX_COMPILER_FLAG("-march=skylake-avx512" COMPILER_SUPPORTS_MARCH_SKYLAKE_AVX512)
if ( COMPILER_SUPPORTS_MARCH_SKYLAKE_AVX512 )
	set_property(SOURCE IQOLanczosResizerImpl_AVX512.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=skylake-avx512")
endif()
