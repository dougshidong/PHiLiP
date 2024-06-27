set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Sod Shock Tube test
# =======================================
configure_file(1D_sod_shock_tube.prm 1D_sod_shock_tube.prm COPYONLY)
add_test(
  NAME 1D_SOD_SHOCK_TUBE_TEST
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_sod_shock_tube.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2D_sod_shock_tube.prm 2D_sod_shock_tube.prm COPYONLY)
add_test(
  NAME 2D_SOD_SHOCK_TUBE_TEST
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_sod_shock_tube.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(3D_sod_shock_tube.prm 3D_sod_shock_tube.prm COPYONLY)
add_test(
  NAME 3D_SOD_SHOCK_TUBE_TEST
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_sod_shock_tube.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
