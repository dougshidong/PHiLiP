configure_file(2d_euler_naca0012_subsonic_05_200.prm 2d_euler_naca0012_subsonic_05_200.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_naca0012_transonic_08_125.prm 2d_euler_naca0012_transonic_08_125.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_TRANSONIC_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_transonic_08_125.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


set (filename "naca0012_hopw_ref2.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(SEND_ERROR
"Missing NACA0012 files named ${filename}. Please download them from
    https://drive.google.com/drive/folders/182JusbWV6NAA8ws1-TTg7M2GLc5jt6_r?usp=sharing
and place them in
      ${CMAKE_CURRENT_SOURCE_DIR}"
      )
endif()
configure_file(naca0012_hopw_ref2.msh naca0012.msh COPYONLY)

#configure_file(naca0012.geo naca0012.geo COPYONLY)
#configure_file(naca0012_noTE.geo naca0012_noTE.geo COPYONLY)
#set(TAR_NAME naca0012_airfoil_Q3)
#add_custom_target(${TAR_NAME}
#    #WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski
#    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#    COMMAND gmsh naca0012_noTE.geo -
#    DEPENDS naca0012_noTE.geo
#    BYPRODUCTS naca0012.msh
#    #COMMAND mv
#    #     "${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski/new_msh41.msh"
#    #     "${CMAKE_BINARY_DIR}/tests/integration_tests_control_files/euler_integration/naca0012/joukowski_R${ref}_Q${order}.msh"
#)
#add_dependencies(grids ${TAR_NAME})
#add_dependencies(2D ${TAR_NAME})
