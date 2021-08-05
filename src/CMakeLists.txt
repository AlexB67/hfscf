project (hfscf LANGUAGES CXX)

# add custom directory to look for .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules )

find_package(Eigen3 3.3.6 REQUIRED NO_MODULE)
find_package(Boost 1.58.0 REQUIRED)
#find_package(libmsym REQUIRED 0.2.4)
#find_package(MKL REQUIRED)
find_package(PkgConfig REQUIRED)
if (PkgConfig_FOUND)
  pkg_check_modules(pc_tclap REQUIRED tclap)
endif()

# Check for OpenMP
find_package(OpenMP)

if (OPENMP_FOUND)
    option(HAS_OPENMP "OpenMP enabled" ON)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    set(HAS_OPENMP OFF)
endif()

#include_directories(irc ${Boost_INCLUDE_DIRS})

include_directories(irc ${Boost_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/subprojects/libmsym 
                   ${CMAKE_SOURC_DIR}/subprojects/libmsym/src) # MKL add /usr/local/intel/mkl/include

# -Wconversion during dev work if you want, note that it generates loads of (harmless) warnings
if(MSVC) # windows not supported anymore
	add_compile_options(/W2  /std:c++17  /openmp[:experimental] /arch:AVX2 /DHAS_CMAKE_CONFIG)
    add_definitions(-DEIGEN_MATRIX_PLUGIN="irc/external/eigen/plugins/Matrix_initializer_list.h") 
else()
    add_definitions(-std=c++17)
    add_definitions(-march=native)
    add_definitions(-msse2)
    add_definitions(-DEIGEN_MATRIX_PLUGIN="irc/external/eigen/plugins/Matrix_initializer_list.h") 
    # boost flag is temporary 'till fixed in boost
    # vla for libmsym temporary
    add_compile_options(-DHAS_CMAKE_CONFIG -DHAVE_EIGEN3 -DBOOST_ALLOW_DEPRECATED_HEADERS -Wall -Wshadow -Wextra -Wpedantic
                        -fno-math-errno -mfma -mavx -mavx -msse4 -fopenmp -fopenmp -fopenmp-simd)
endif()                  # DHAS_CMAKE_CONFIG indicates we are using a cmake config since we also support meson
                         # use -DEIGEN_USE_MKL_ALL with mkl 
                         # DBOOST_ALLOW_DEPRECATED_HEADERS shut up boost, should be fixed in next version

set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set include folders
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_BINARY_DIR}
                    ${CPPUNIT_INCLUDE_DIR}
                    ${EIGEN_INCLUDE_DIRS})

# Add sources
file(GLOB SOURCES   "*.cpp")
file(GLOB SOURCES1  "hfscf/*.cpp")
file(GLOB SOURCES2  "settings/*.cpp")
file(GLOB SOURCES3  "molecule/*.cpp")
file(GLOB SOURCES4  "math/*.cpp")
file(GLOB SOURCES5  "integrals/*.cpp")
file(GLOB SOURCES6  "basis/*.cpp")
file(GLOB SOURCES7  "utils/*.cpp")
file(GLOB SOURCES8  "postscf/*.cpp")
file(GLOB SOURCES9  "config_interaction/*.cpp")
file(GLOB SOURCES10 "irc/*.cpp")
file(GLOB SOURCES11 "mol_properties/*.cpp")
file(GLOB SOURCES12 "gradient/*.cpp")
file(GLOB SOURCES13 "hessian/*.cpp")
file(GLOB SOURCES14 "scfsolver/*.cpp")
file(GLOB SOURCES15 "symmetry/*.cpp")

# Set executable
add_executable(hfscf ${SOURCES}   ${SOURCES1}  ${SOURCES2}  ${SOURCES3}  ${SOURCES4} 
                     ${SOURCES5}  ${SOURCES6}  ${SOURCES7}  ${SOURCES8}  ${SOURCES9}
                     ${SOURCES10} ${SOURCES11} ${SOURCES12} ${SOURCES13} ${SOURCES14} ${SOURCES15})

				 
target_link_libraries (hfscf Eigen3::Eigen msym ${Boost_LIBRARIES}) # -lcpuid for cpu detection support

target_include_directories (hfscf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
                            ${CMAKE_SOURCE_DIR}/subprojects/libmsym/src)

#mkl works. Add or whatever location you have
#-L/usr/local/intel/mkl/lib/intel64
#"-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")

