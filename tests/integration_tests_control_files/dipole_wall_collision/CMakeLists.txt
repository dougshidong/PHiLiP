set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Dipole Wall Collision (DWC)
# =======================================
# ----------------------------------------
# -- Reference for flow setup: 
# -- Julien Brillon's thesis and the references therein
# ----------------------------------------
configure_file(viscous_DWC_ILES_NSFR_cHU_IR_2PF_GL_OI-0_dofs0384_p5.prm viscous_DWC_ILES_NSFR_cHU_IR_2PF_GL_OI-0_dofs0384_p5.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_DIPOLE_WALL_COLLISION_QUANTITY_CHECK_STRONG_DG_P5
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_DWC_ILES_NSFR_cHU_IR_2PF_GL_OI-0_dofs0384_p5.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_VISCOUS_DIPOLE_WALL_COLLISION_QUANTITY_CHECK_STRONG_DG_P5 DIPOLE_WALL_COLLISION_INTEGRATION
                                                                               2D
                                                                               PARALLEL
                                                                               NAVIER_STOKES
                                                                               WALL_BOUNDARY_CONDITION
                                                                               RUNGE-KUTTA
                                                                               STRONG
                                                                               UNCOLLOCATED
                                                                               EXTRA-LONG
                                                                               INTEGRATION_TEST)
# ----------------------------------------
configure_file(viscous_DWC_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs0192_p2.prm viscous_DWC_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs0192_p2.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_DIPOLE_WALL_COLLISION_QUANTITY_CHECK_STRONG_DG_P2
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_DWC_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs0192_p2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_VISCOUS_DIPOLE_WALL_COLLISION_QUANTITY_CHECK_STRONG_DG_P2 DIPOLE_WALL_COLLISION_INTEGRATION
                                                                               2D
                                                                               PARALLEL
                                                                               NAVIER_STOKES
                                                                               WALL_BOUNDARY_CONDITION
                                                                               RUNGE-KUTTA
                                                                               STRONG
                                                                               UNCOLLOCATED
                                                                               LONG
                                                                               INTEGRATION_TEST)
# ----------------------------------------
configure_file(viscous_DWC_ILES_NSFR_cHU_IR_2PF_GL_OI-0_dofs0384_p5_quick.prm viscous_DWC_ILES_NSFR_cHU_IR_2PF_GL_OI-0_dofs0384_p5_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_DIPOLE_WALL_COLLISION_QUANTITY_CHECK_STRONG_DG_P5_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_DWC_ILES_NSFR_cHU_IR_2PF_GL_OI-0_dofs0384_p5_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_VISCOUS_DIPOLE_WALL_COLLISION_QUANTITY_CHECK_STRONG_DG_P5_QUICK DIPOLE_WALL_COLLISION_INTEGRATION
                                                                                     2D
                                                                                     PARALLEL
                                                                                     NAVIER_STOKES
                                                                                     RUNGE-KUTTA
                                                                                     STRONG
                                                                                     UNCOLLOCATED
                                                                                     QUICK
                                                                                     INTEGRATION_TEST)

# ----------------------------------------