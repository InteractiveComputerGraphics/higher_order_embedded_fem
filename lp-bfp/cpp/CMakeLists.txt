cmake_minimum_required(VERSION 3.10)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
set(WINDOWS_EXPORT_ALL_SYMBOLS true)

SET(ORTOOLS_ROOT $ENV{ORTOOLS_ROOT})
set(ORTOOLS_INCLUDE ${ORTOOLS_ROOT}/include)
set(ORTOOLS_LIB ${ORTOOLS_ROOT}/lib)

add_library(lp-bfp STATIC lp-bfp.cpp lp-bfp.h)
target_include_directories(lp-bfp SYSTEM PUBLIC ${ORTOOLS_INCLUDE})
target_link_libraries(lp-bfp libortools)

# Ideally we'd use target_link_directories, but it is only supported by fairly recent versions of CMake
link_directories(${ORTOOLS_LIB})

IF (WIN32)
    # OR-Tools wants to define min/max functions but also includes windows.h which defines min/max macros itself
    # Therefore add a flag that disables the min/max macros from windows.h
    target_compile_definitions(lp-bfp PUBLIC NOMINMAX)
ENDIF()

install(TARGETS lp-bfp
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)