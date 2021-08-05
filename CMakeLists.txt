cmake_minimum_required (VERSION 3.0)

SET($(PROJECT_NAME) "hfscf")
project ($(PROJECT_NAME))

# config file
SET(VERSION_MAJOR "0")
SET(VERSION_MINOR "7")
SET(VERSION_MICRO "1")
SET(HFSCF_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
SET(HFSCF_COMPILER "${CMAKE_CXX_COMPILER_VERSION}")
configure_file(hfscf_cmake_config.h.in cmakeconfig.h )


# Enable release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(subprojects/libmsym)
add_subdirectory(src)
