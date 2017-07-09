add_library(iqo IQOHWCap.cpp
                IQOLanczosResizer.cpp
                IQOLanczosResizerImpl_Generic.cpp
                IQOLanczosResizerImpl_SSE4_1.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
