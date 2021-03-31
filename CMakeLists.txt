cmake_minimum_required(VERSION 3.10)
project(PictureSimilarity)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_LIB_PATH: ${OpenCV_LIB_PATH}")

add_executable(PictureSimilarity main.cpp)
target_include_directories(PictureSimilarity PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PictureSimilarity PRIVATE ${OpenCV_LIBS})