cmake_minimum_required(VERSION 3.16)
project(high_precision_felsenstein LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_library(QD_LIBRARY qd REQUIRED)

add_executable(high_precision_felsenstein
  main.cpp
  felsenstein_qd.cpp
)

target_include_directories(high_precision_felsenstein PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(high_precision_felsenstein PRIVATE ${QD_LIBRARY})
