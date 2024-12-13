set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Low Density Test - Euler
# =======================================

#configure_file(2D_low_density.prm 2D_low_density.prm COPYONLY)
#add_test(
#  NAME 2D_LOW_DENSITY_EULER_TEST
#  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_low_density.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(2D_LOW_DENSITY_EULER_TEST LOW_DENSITY_2D
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            LONG
#                                            INTEGRATION_TEST)

#configure_file(2D_low_density_OOA.prm 2D_low_density_OOA.prm COPYONLY)
#add_test(
#  NAME 2D_LOW_DENSITY_EULER_OOA_TEST
#  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_low_density_OOA.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(2D_LOW_DENSITY_EULER_OOA_TEST LOW_DENSITY_2D
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            CONVERGENCE
#                                            LONG
#                                            INTEGRATION_TEST)

#configure_file(2D_low_density_OOA_NSFR.prm 2D_low_density_OOA_NSFR.prm COPYONLY)
#add_test(
#  NAME 2D_LOW_DENSITY_EULER_OOA_NSFR_TEST
#  COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_low_density_OOA_NSFR.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(2D_LOW_DENSITY_EULER_OOA_NSFR_TEST LOW_DENSITY_2D
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG-SPLIT
#                                            COLLOCATED
#                                            LIMITER
#                                            CONVERGENCE
#                                            LONG
#                                            INTEGRATION_TEST)