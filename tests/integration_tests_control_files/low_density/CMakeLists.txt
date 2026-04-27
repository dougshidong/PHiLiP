set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Low Density Test - Euler
# =======================================

 configure_file(1D_low_density_OOA_NSFR.prm 1D_low_density_OOA_NSFR.prm COPYONLY)
 add_test(
   NAME 1D_LOW_DENSITY_EULER_OOA_NSFR_TEST
   COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_low_density_OOA_NSFR.prm
   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
 )
 set_tests_labels(1D_LOW_DENSITY_EULER_OOA_NSFR_TEST    LOW_DENSITY
                                                        1D
                                                        SERIAL
                                                        EULER
                                                        RUNGE-KUTTA
                                                        STRONG-SPLIT
                                                        COLLOCATED
                                                        LIMITER
                                                        MODERATE
                                                        CONVERGENCE
                                                        INTEGRATION_TEST)

#configure_file(2D_low_density.prm 2D_low_density.prm COPYONLY)
#add_test(
#  NAME 2D_LOW_DENSITY_EULER_TEST
#  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_low_density.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(2D_LOW_DENSITY_EULER_TEST LOW_DENSITY
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            QUICK
#                                            INTEGRATION_TEST)

configure_file(2D_low_density_OOA_NSFR.prm 2D_low_density_OOA_NSFR.prm COPYONLY)
add_test(
  NAME 2D_LOW_DENSITY_EULER_OOA_NSFR_TEST
  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_low_density_OOA_NSFR.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_LOW_DENSITY_EULER_OOA_NSFR_TEST LOW_DENSITY
                                            2D
                                            PARALLEL
                                            EULER
                                            RUNGE-KUTTA
                                            STRONG-SPLIT
                                            COLLOCATED
                                            LIMITER
                                            CONVERGENCE
                                            LONG
                                            INTEGRATION_TEST)
