CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

# Set the name of the project and target:
#SET(TARGET "PHiLiP")

set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

SET(CLEAN_UP_FILES sol.* grid.*)

# Find deal.ii library
find_package(deal.II 9.0.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

SET(CMAKE_BUILD_TYPE Release)
# libraries to be compiled

DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(philip)
add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

