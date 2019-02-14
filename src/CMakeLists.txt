#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#add_library(advection_lib
#    advection_boundary.h
#    advection_explicit.cpp 
#    advection_explicit.h)
#target_include_directories(advection_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
add_library(advection_lib
    parameters.cpp
    assemble_explicit.cpp 
    dg.cpp )
target_include_directories(advection_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

DEAL_II_SETUP_TARGET(advection_lib)

# The main program
#add_executable(PHiLiP main.cpp)
SET(TARGET "PHiLiP")
SET(TARGET_SRC main.cpp)
DEAL_II_INVOKE_AUTOPILOT()

DEAL_II_SETUP_TARGET(PHiLiP)
target_link_libraries(PHiLiP advection_lib)
