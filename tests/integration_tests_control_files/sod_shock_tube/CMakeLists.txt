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
set_tests_labels(1D_SOD_SHOCK_TUBE_TEST SOD_SHOCK_TUBE
                                        1D
                                        SERIAL
                                        EULER
                                        RUNGE-KUTTA
                                        STRONG-SPLIT
                                        COLLOCATED
                                        LIMITER
                                        QUICK
                                        INTEGRATION_TEST)
