set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Generating 2D unstructured NACA0012 mesh...")
string(CONCAT GMSH_GEO naca0012_noTE.geo)
configure_file(${GMSH_GEO} ${GMSH_GEO} COPYONLY)

execute_process(COMMAND gmsh ${GMSH_GEO} -
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GMSH_RESULT
                OUTPUT_QUIET)
if(NOT GMSH_RESULT EQUAL "0")
    message(FATAL_ERROR
            "gmsh ${GMSH_RESULT}, please check ${GMSH_GEO} file")
endif()

string(CONCAT GMSH_MSH naca0012.msh)
configure_file(${GMSH_MSH} ${GMSH_MSH} COPYONLY)

configure_file(2d_euler_naca_optimization.prm 2d_euler_naca_optimization.prm COPYONLY)
configure_file(naca0012.msh naca0012.msh COPYONLY)
add_test(
  NAME 2D_EULER_NACA0012_OPTIMIZATION_CONSTRAINED
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca_optimization.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
