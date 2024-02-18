set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Generating 3D unstructured NACA0012 mesh...")
string(CONCAT GMSH_GEO naca0012_wing_unstructured_cutoff.geo)
configure_file(${GMSH_GEO} ${GMSH_GEO} COPYONLY)

execute_process(COMMAND gmsh ${GMSH_GEO} -
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GMSH_RESULT
                OUTPUT_QUIET)
if(NOT GMSH_RESULT EQUAL "0")
    message(FATAL_ERROR
            "gmsh ${GMSH_RESULT}, please check ${GMSH_GEO} file")
endif()

string(CONCAT GMSH_MSH naca0012_wing_unstructured_cutoff.msh)
configure_file(${GMSH_MSH} ${GMSH_MSH} COPYONLY)

configure_file(3d_euler_naca_opt.prm 3d_euler_naca_opt.prm COPYONLY)
add_test(
  NAME 3D_EULER_NACA0012_OPTIMIZATION_CONSTRAINED
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_euler_naca_opt.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
