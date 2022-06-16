set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Viscous Taylor Green Vortex (TGV)
# =======================================
# ----------------------------------------
# -- Reference for flow setup: 
# -- -- Diosady, L., and S. Murman. "Case 3.3: Taylor green vortex evolution." Case Summary for 3rd International Workshop on Higher-Order CFD Methods. 2015.
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_quick.prm viscous_taylor_green_vortex_energy_check_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_QUICK_WEAK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_quick_strong.prm viscous_taylor_green_vortex_energy_check_quick_strong.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_QUICK_STRONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_quick_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_long.prm viscous_taylor_green_vortex_energy_check_long.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_long.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_restart_check.prm viscous_taylor_green_vortex_restart_check.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_CHECK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_restart_check.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_restart_from_parameter_file_check.prm viscous_taylor_green_vortex_restart_from_parameter_file_check.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_FROM_PARAMETER_FILE_CHECK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_restart_from_parameter_file_check.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
