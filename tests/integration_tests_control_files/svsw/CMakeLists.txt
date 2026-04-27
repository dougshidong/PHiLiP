set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Strong Vortex Shock Wave Case
# =======================================
configure_file(2D_svsw.prm 2D_svsw.prm COPYONLY)
add_test(
  NAME 2D_SVSW_TEST
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_svsw.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_SVSW_TEST SVSW_TEST
                                            2D
                                            PARALLEL
                                            EULER
                                            RUNGE-KUTTA
                                            STRONG
                                            COLLOCATED
                                            LIMITER
                                            MODERATE
                                            INTEGRATION_TEST)
