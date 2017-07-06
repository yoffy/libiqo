add_library(iqo IQOLanczosResizer.cpp
                IQOLanczosResizerImpl_Generic.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
