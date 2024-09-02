cmake_minimum_required(VERSION 3.2...3.18)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

include_directories(${OpenCV_INCLUDE_DIR})

file(GLOB APRILTAG_SRCS "src/*.cc")
include_directories(include)

# include_directories(../../include)
# include_directories(${CMAKE_SOURCE_DIR})
# include_directories(${CMAKE_SOURCE_DIR}/lib)
# include_directories(${CMAKE_SOURCE_DIR}/lib/thirdparty/Sophus)

add_library(ethz_apriltag STATIC ${APRILTAG_SRCS})

target_include_directories(ethz_apriltag PUBLIC include)
target_link_libraries(ethz_apriltag PUBLIC caliber::opencv Sophus::Sophus)