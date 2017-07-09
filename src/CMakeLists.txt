add_library(iqo IQOHWCap.cpp
                IQOLanczosResizer.cpp
                IQOLanczosResizerImpl_Generic.cpp
                IQOLanczosResizerImpl_SSE4_1.cpp
                IQOLanczosResizerImpl_AVX2FMA.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_property(SOURCE IQOLanczosResizerImpl_SSE4_1.cpp  APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 -msse4.1")
set_property(SOURCE IQOLanczosResizerImpl_AVX2FMA.cpp APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core-avx2")
