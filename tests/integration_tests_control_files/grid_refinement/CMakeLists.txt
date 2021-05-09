set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Other testing (anisotropic, smoothness, etc.)
configure_file(2d_anisotropic_ff_h.prm 2d_anisotropic_ff_h.prm COPYONLY)
add_test(
  NAME 2D_ANISOTROPIC_FF_H
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_anisotropic_ff_h.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# sizefield write tests
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

# anisotropic gmsh grid refinement studies

if(${ENABLE_GMSH})
  # S-Shock (p=1, p=2)
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

  # Boundary Layer (p=2, p=3)
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
endif(${ENABLE_GMSH})

# Adjoint grid refinement studies

if(${ENABLE_GMSH})
  # S-Shock (p=1, p=2)
  configure_file(2d_gmsh_adjoint_sshock_p1.prm 2d_gmsh_adjoint_sshock_p1.prm COPYONLY)
  add_test(
    NAME 2D_GMSH_ADJOINT_SSHOCK_P1
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_adjoint_sshock_p1.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )

  configure_file(2d_gmsh_adjoint_sshock_p2.prm 2d_gmsh_adjoint_sshock_p2.prm COPYONLY)
  add_test(
    NAME 2D_GMSH_ADJOINT_SSHOCK_P2
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_adjoint_sshock_p2.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )

  # Boundary Layer (p=2, p=3)
  configure_file(2d_gmsh_adjoint_boundarylayer_p2.prm 2d_gmsh_adjoint_boundarylayer_p2.prm COPYONLY)
  add_test(
    NAME 2D_GMSH_ADJOINT_BOUNDARYLAYER_P2
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer_p2.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )

  configure_file(2d_gmsh_adjoint_boundarylayer_p3.prm 2d_gmsh_adjoint_boundarylayer_p3.prm COPYONLY)
  add_test(
    NAME 2D_GMSH_ADJOINT_BOUNDARYLAYER_P3
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer_p3.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )
endif(${ENABLE_GMSH})

