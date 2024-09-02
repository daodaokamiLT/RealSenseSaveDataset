project(gtloader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(realsense2 REQUIRED)
if(realsense2_FOUND)
    message(STATUS "Found realsense:")
    message(STATUS "realsense2_INCLUDE_DIR ${realsense2_INCLUDE_DIR}")
    message(STATUS "realsense2_LIBRARY ${realsense2_LIBRARY}")
    list(APPEND EXTERNAL_LIBRARIES ${realsense2_LIBRARY})
else()
    message(STATUS "Cannot found realsense")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 QUIET)
if(NOT TARGET glfw)
    message(STATUS "GLFW 3.3 not found; using internal version")
    set(GLFW_INSTALL ON CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/glfw)
else()
message(STATUS "GLFW 3.3 found")

message(glfw ${OPENGL_LIBRARIES})
endif()

find_package(OpenCV REQUIRED)
include_directories({OpenCV_INCLUDE_DIRS})

add_executable(running_datagrabber running_datagrabber.cpp)
target_link_libraries(running_datagrabber ${EXTERNAL_LIBRARIES} glfw ${OPENGL_LIBRARIES}
                        ${OpenCV_LIBS})