cmake_minimum_required(VERSION 2.6)

project(blur_test)

list (APPEND CMAKE_CXX_FLAGS "-std=c++1y")

add_executable(blur_test 
    main.cpp
)

target_link_libraries (blur_test
    OpenCL
    X11
    pthread
)

install(TARGETS blur_test RUNTIME DESTINATION bin)
