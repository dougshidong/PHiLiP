set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(2d_anisotropic_ff_h.prm 2d_anisotropic_ff_h.prm COPYONLY)
add_test(
  NAME 2D_ANISOTROPIC_FF_H
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_anisotropic_ff_h.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_msh_size_field.prm 2d_msh_size_field.prm COPYONLY)
add_test(
  NAME 2D_MSH_SIZE_FIELD_H
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_msh_size_field.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_msh_quadratic_field.prm 2d_msh_quadratic_field.prm COPYONLY)
add_test(
  NAME 2D_MSH_QUADRATIC_FIELD_H
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_msh_quadratic_field.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_gmsh_aniso_sshock_p1.prm 2d_gmsh_aniso_sshock_p1.prm COPYONLY)
add_test(
  NAME 2D_GMSH_ANISO_SSHOCK_P1
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_sshock_p1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_gmsh_aniso_sshock_p2.prm 2d_gmsh_aniso_sshock_p2.prm COPYONLY)
add_test(
  NAME 2D_GMSH_ANISO_SSHOCK_P2
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_sshock_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_gmsh_aniso_boundarylayer_p2.prm 2d_gmsh_aniso_boundarylayer_p2.prm COPYONLY)
add_test(
  NAME 2D_GMSH_ANISO_BOUNDARYLAYER_P2
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_gmsh_aniso_boundarylayer_p2.prm 2d_gmsh_aniso_boundarylayer_p2.prm COPYONLY)
add_test(
  NAME 2D_GMSH_ANISO_BOUNDARYLAYER_P3
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_gmsh_adjoint_sshock.prm 2d_gmsh_adjoint_sshock.prm COPYONLY)
add_test(
  NAME 2D_GMSH_ADJOINT_SSHOCK
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_adjoint_sshock.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_gmsh_adjoint_boundarylayer.prm 2d_gmsh_adjoint_boundarylayer.prm COPYONLY)
add_test(
  NAME 2D_GMSH_ADJOINT_BOUNDARYLAYER
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
