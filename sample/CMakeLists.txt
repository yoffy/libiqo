add_executable(resize_yuv420p resize_yuv420p.cpp)

target_link_libraries(resize_yuv420p LINK_PUBLIC iqo)

if ( CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98")
elseif ( MSVC )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _CRT_SECURE_NO_WARNINGS=1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG")
endif()