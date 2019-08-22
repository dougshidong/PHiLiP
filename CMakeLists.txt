project(PHiLiP)

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
    
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror")

set(MPIMAX 4 CACHE STRING "Default number of processors used in ctest mpirun -np MPIMAX. Not the same as ctest -jX")


#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -Og -g")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CLEAN_UP_FILES ./bin/* ./CMakeCache.txt)

# Find deal.ii library
find_package(deal.II 9.1.1 QUIET
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Make sure deal.II has the proper external dependencies
IF(NOT ${DEAL_II_WITH_MPI})
  message(FATAL_ERROR "\n" "*** deal.II needs to be configured with -DDEAL_II_WITH_MPI=ON ***\n\n")
ENDIF()
IF(NOT ${DEAL_II_WITH_TRILINOS})
  message(FATAL_ERROR "\n" "*** deal.II needs to be configured with -DDEAL_II_WITH_TRILINOS=ON ***\n\n")
ENDIF()
IF(NOT ${DEAL_II_WITH_P4EST})
  message(FATAL_ERROR "\n" "*** deal.II needs to be configured with -DDEAL_II_WITH_P4EST=ON ***\n\n")
ENDIF()
IF(NOT ${DEAL_II_WITH_METIS})
  message(FATAL_ERROR "\n" "*** deal.II needs to be configured with -DDEAL_II_WITH_METIS=ON ***\n\n")
ENDIF()

# Use ld.gold for faster linking
option(USE_LD_GOLD "Use GNU gold linker" ON)
if(USE_LD_GOLD AND "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version OUTPUT_VARIABLE stdout ERROR_QUIET)
  if("${stdout}" MATCHES "GNU gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
  else()
    message(WARNING "GNU gold linker isn't available, using the default system linker.")
  endif()
endif()

add_custom_target(1D COMMAND ${CMAKE_COMMAND} -E echo "Makes all 1D executables, including tests. Allows ctest -R 1D")
add_custom_target(2D COMMAND ${CMAKE_COMMAND} -E echo "Makes all 2D executables, including tests. Allows ctest -R 2D")
add_custom_target(3D COMMAND ${CMAKE_COMMAND} -E echo "Makes all 3D executables, including tests. Allows ctest -R 3D")
SET(_dimension_targets "#
#      make 1D             - to build all 1D executables, including tests, allow 'ctest -R 1D' 
#      make 2D             - to build all 2D executables, including tests, allow 'ctest -R 2D'
#      make 3D             - to build all 3D executables, including tests, allow 'ctest -R 3D' ")
SET(_philip_targets "#
#      make PHiLiP_1D      - to build main program (wihtout tests) in 1D
#      make PHiLiP_2D      - to build main program (wihtout tests) in 2D
#      make PHiLiP_3D      - to build main program (wihtout tests) in 3D ")
add_custom_target(unit_tests)

# Source code
include_directories(src)
add_subdirectory(src)

# Test directory
enable_testing()
add_subdirectory(tests)

# Documentation
add_subdirectory(doc)

##########
# Add a few commands to make life easier

#include(ProcessorCount)
#ProcessorCount(NPROC)

# Define custom targets to easily switch the build type:
add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )
IF(${DEAL_II_BUILD_TYPE} MATCHES "DebugRelease")
SET(_switch_targets "#
#      make debug          - to switch the build type to 'Debug'
#      make release        - to switch the build type to 'Release' "
)
ENDIF()

# Print out some usage information to file:
FILE(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
"MESSAGE(
\"###
#
#  Project  ${TARGET}  set up with  ${DEAL_II_PACKAGE_NAME}-${DEAL_II_PACKAGE_VERSION}  found at
#      ${DEAL_II_PATH}
#
#  CMAKE_BUILD_TYPE:          ${CMAKE_BUILD_TYPE}
#
#  You can now run
#      make                - to compile and link all the program and tests.
${_philip_targets}
${_dimension_targets}
${_switch_targets}
")
  FILE(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
"#
#      make clean          - to remove the generated executable as well as
#                               all intermediate compilation files
#      make info           - to view this message again
#
#  Note that you may append -j N, where N represents the number of processors
#  to compile with. If you are compiling on 8GB of memory, be careful not to
#  use too many processors, as you will run out of RAM. If you have (m) GB of 
#  available memory, you can use around (m-1) processor without having to use the swap.
#
###\")"
)

add_custom_target(info
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
)

# Print this message once:
#IF(NOT USAGE_PRINTED)
#  INCLUDE(${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake)
#  SET(USAGE_PRINTED TRUE CACHE INTERNAL "")
#ELSE()
#  MESSAGE(STATUS "Run  make info  to print a detailed help message")
#ENDIF()
INCLUDE(${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake)
MESSAGE(STATUS "Run  make info  to print a detailed help message")

MESSAGE(STATUS "
###
#
# Using MPIMAX = ${MPIMAX}, which sets the default values used by ctest for the MPI runs.
# Can use cmake ../ -DMPIMAX=XX to change this default value.
#
###
")
