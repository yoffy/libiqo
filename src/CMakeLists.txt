add_library(iqo IQOLanczosResizer.cpp
                IQOLanczosResizer_Generic.cpp)
target_include_directories(iqo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
