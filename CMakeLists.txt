cmake_minimum_required(VERSION 2.8)
set(ProjName caffe-onclick)
project(${ProjName})
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Caffe  REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})

add_executable(evaluation cpp/evaluation.cpp cpp/LenetClassifier.cpp)
target_link_libraries(evaluation ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable(cpp4caffe cpp4caffe/evaluation.cpp)
target_link_libraries(cpp4caffe ${OpenCV_LIBS} ${Caffe_LIBRARIES})