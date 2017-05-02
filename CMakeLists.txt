project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources
   src/ukf.cpp
   src/main.cpp
   src/tools.cpp)

set(tests
  src/ukf.cpp
  src/tools.cpp
  src/main_test.cpp)

add_executable(UnscentedKF ${sources})
add_executable(UnscentedKF_test ${tests})
