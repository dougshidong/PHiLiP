project(PHiLiP)

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_${BUILD_DIM}D)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

set(CLEAN_UP_FILES ./bin/* ./CMakeCache.txt)

# Find deal.ii library
find_package(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(doc)
