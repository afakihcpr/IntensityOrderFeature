cmake_minimum_required(VERSION 3.1)
project(iop)

set (CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR $ENV{HOME}/libs/opencv/install/share/OpenCV/)

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

include_directories(
    include
)

add_library(iop_features SHARED
    include/Common.h
    include/MyDescriptors.h
    include/Utils.h
    src/Utils.cpp
    src/MyDescriptors.cpp
)
target_link_libraries(iop_features ${OpenCV_LIBS})
set_target_properties(iop_features PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(test_iop
    src/Main.cpp
)
target_link_libraries(test_iop iop_features ${OpenCV_LIBS})

export(TARGETS iop_features
    FILE "${PROJECT_BINARY_DIR}/IopTargets.cmake")

export(PACKAGE Iop)

INSTALL(TARGETS iop_features
DESTINATION lib/)

INSTALL(FILES 
	include/MyDescriptors.h
	include/Common.h
	include/Utils.h
        DESTINATION include/iop_features/)

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/IopConfig.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/IopConfig.cmake
  @ONLY
)
INSTALL(
    FILES ${PROJECT_BINARY_DIR}/cmake/IopConfig.cmake
    DESTINATION share/cmake
)
