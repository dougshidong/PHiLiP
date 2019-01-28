project(eulerflow)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
add_library(eulerflow_lib
    EulerFlow.cpp 
    EulerFlow.h)
target_include_directories(eulerflow_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

DEAL_II_SETUP_TARGET(eulerflow_lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
add_library(advection_lib
    advection.cpp 
    advection.h)
target_include_directories(advection_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

DEAL_II_SETUP_TARGET(advection_lib)

# The main program
add_executable(PHiLiP main.cpp)

DEAL_II_SETUP_TARGET(PHiLiP)
target_link_libraries(PHiLiP eulerflow_lib)
