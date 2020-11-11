configure_file(2d_euler_naca0012.prm 2d_euler_naca0012.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_NACA0012_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(naca0012.geo naca0012.geo COPYONLY)
set(TAR_NAME naca0012_airfoil_Q3)
add_custom_target(${TAR_NAME}
    #WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    COMMAND gmsh naca0012.geo -
    #COMMAND mv
    #     "${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski/new_msh41.msh"
    #     "${CMAKE_BINARY_DIR}/tests/integration_tests_control_files/euler_integration/naca0012/joukowski_R${ref}_Q${order}.msh"
)
add_dependencies(grids ${TAR_NAME})
add_dependencies(2D ${TAR_NAME})
