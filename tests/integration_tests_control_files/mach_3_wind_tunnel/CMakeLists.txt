set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Sod Shock Tube test
# =======================================
configure_file(2D_mach_3_wind_tunnel.prm 2D_mach_3_wind_tunnel.prm COPYONLY)
add_test(
  NAME 2D_MACH_3_WIND_TUNNEL
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_mach_3_wind_tunnel.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
