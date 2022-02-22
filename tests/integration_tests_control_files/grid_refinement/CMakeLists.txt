set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Other testing (anisotropic, smoothness, etc.)
configure_file(2d_anisotropic_ff_h.prm 2d_anisotropic_ff_h.prm COPYONLY)
add_test(
  NAME 2D_ANISOTROPIC_FF_H
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_anisotropic_ff_h.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# sizefield write tests
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
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer_p2.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )

  configure_file(2d_gmsh_aniso_boundarylayer_p3.prm 2d_gmsh_aniso_boundarylayer_p3.prm COPYONLY)
  add_test(
    NAME 2D_GMSH_ANISO_BOUNDARYLAYER_P3
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_aniso_boundarylayer_p3.prm
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
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_adjoint_boundarylayer_p2.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )

  configure_file(2d_gmsh_adjoint_boundarylayer_p3.prm 2d_gmsh_adjoint_boundarylayer_p3.prm COPYONLY)
  add_test(
    NAME 2D_GMSH_ADJOINT_BOUNDARYLAYER_P3
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gmsh_adjoint_boundarylayer_p3.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )
endif(${ENABLE_GMSH})

# Lp-CVT tests (single step, full loop not availible)
# Reads input grid, runs single step, outputs new metric 2d_msh_size_field
# Copies input mesh and expected output (not regression tested as this varies with GMSH versions)
# .msh input/output naming convention in relation to Lp-CVT (reversed here)

# all cases started on 16x16 regular grid of chosen polynomial order and ran 3 steps with 1.5x complexity growth
# Note the numbering is based on remeshing iterations. e.g. 
# PHiLiP, initial grid -> input_*_0.msh
# LpCVT, input_*_0.msh -> output_*_0.msh
# PHiLiP, output_*_0.msh -> input_*_1.msh... etc.

# Feature-based S-Shock (p=1, p=2)
configure_file(msh_in/output_ss15_fb_p1_2.msh msh_in/output_ss15_fb_p1_2.msh COPYONLY)
configure_file(msh_out/input_ss15_fb_p1_3.msh msh_out/input_ss15_fb_p1_3.msh COPYONLY)
configure_file(2d_lpcvt_fb_sshock_p1.prm 2d_lpcvt_fb_sshock_p1.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_FB_SSHOCK_P1
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_fb_sshock_p1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(msh_in/output_ss15_fb_p2_2.msh msh_in/output_ss15_fb_p2_2.msh COPYONLY)
configure_file(msh_out/input_ss15_fb_p2_3.msh msh_out/input_ss15_fb_p2_3.msh COPYONLY)
configure_file(2d_lpcvt_fb_sshock_p2.prm 2d_lpcvt_fb_sshock_p2.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_FB_SSHOCK_P2
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_fb_sshock_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Feature-based Boundary Layer (p=2, p=3)
configure_file(msh_in/output_bl15_fb_p2_2.msh msh_in/output_bl15_fb_p2_2.msh COPYONLY)
configure_file(msh_out/input_bl15_fb_p2_3.msh msh_out/input_bl15_fb_p2_3.msh COPYONLY)
configure_file(2d_lpcvt_fb_boundarylayer_p2.prm 2d_lpcvt_fb_boundarylayer_p2.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_FB_BOUNDARYLAYER_P2
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_fb_boundarylayer_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(msh_in/output_bl15_fb_p3_2.msh msh_in/output_bl15_fb_p3_2.msh COPYONLY)
configure_file(msh_out/input_bl15_fb_p3_3.msh msh_out/input_bl15_fb_p3_3.msh COPYONLY)
configure_file(2d_lpcvt_fb_boundarylayer_p3.prm 2d_lpcvt_fb_boundarylayer_p3.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_FB_BOUNDARYLAYER_P3
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_fb_boundarylayer_p3.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


# Adjoint-based S-Shock (p=1, p=2)
configure_file(msh_in/output_ss15_adj_p1_2.msh msh_in/output_ss15_adj_p1_2.msh COPYONLY)
configure_file(msh_out/input_ss15_adj_p1_3.msh msh_out/input_ss15_adj_p1_3.msh COPYONLY)
configure_file(2d_lpcvt_adj_sshock_p1.prm 2d_lpcvt_adj_sshock_p1.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_ADJ_SSHOCK_P1
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_adj_sshock_p1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(msh_in/output_ss15_adj_p2_2.msh msh_in/output_ss15_adj_p2_2.msh COPYONLY)
configure_file(msh_out/input_ss15_adj_p2_3.msh msh_out/input_ss15_adj_p2_3.msh COPYONLY)
configure_file(2d_lpcvt_adj_sshock_p2.prm 2d_lpcvt_adj_sshock_p2.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_ADJ_SSHOCK_P2
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_adj_sshock_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Adjoint-based Boundary Layer (p=2, p=3)
configure_file(msh_in/output_bl15_adj_p2_2.msh msh_in/output_bl15_adj_p2_2.msh COPYONLY)
configure_file(msh_out/input_bl15_adj_p2_3.msh msh_out/input_bl15_adj_p2_3.msh COPYONLY)
configure_file(2d_lpcvt_adj_boundarylayer_p2.prm 2d_lpcvt_adj_boundarylayer_p2.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_ADJ_BOUNDARYLAYER_P2
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_adj_boundarylayer_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(msh_in/output_bl15_adj_p3_2.msh msh_in/output_bl15_adj_p3_2.msh COPYONLY)
configure_file(msh_out/input_bl15_adj_p3_3.msh msh_out/input_bl15_adj_p3_3.msh COPYONLY)
configure_file(2d_lpcvt_adj_boundarylayer_p3.prm 2d_lpcvt_adj_boundarylayer_p3.prm COPYONLY)
add_test(
  NAME 2D_LPCVT_ADJ_BOUNDARYLAYER_P3
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_lpcvt_adj_boundarylayer_p3.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Dual Weighted Residual and Convergence Tests
configure_file(2d_sshock_dual_weighted_residual_convergence.prm 2d_sshock_dual_weighted_residual_convergence.prm  COPYONLY)
add_test(
  NAME MPI_2D_DUAL_WEIGHTED_RESIDUAL_CONVERGENCE_SSHOCK
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_sshock_dual_weighted_residual_convergence.prm 
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
