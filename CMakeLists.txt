cmake_minimum_required(VERSION 3.10)
project(PictureSimilarity)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(PictureSimilarity main.cpp)
target_include_directories(PictureSimilarity PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PictureSimilarity PRIVATE ${OpenCV_LIBS})