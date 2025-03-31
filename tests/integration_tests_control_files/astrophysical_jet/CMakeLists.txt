set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Astrophysical Mach Jet (Mach 80 or Mach 2000)
# =======================================
# configure_file(2D_astrophysical_jet.prm 2D_astrophysical_jet.prm COPYONLY)
# add_test(
#   NAME 2D_ASTROPHYSICAL_JET_MACH80
#   COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_astrophysical_jet.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
#set_tests_labels(2D_ASTROPHYSICAL_JET_MACH80 ASTROPHYSICAL_JET
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            EXTRA-LONG
#                                            INTEGRATION_TEST)

# configure_file(2D_astrophysical_jet_mach2000.prm 2D_astrophysical_jet_mach2000.prm COPYONLY)
# add_test(
#   NAME 2D_ASTROPHYSICAL_JET_MACH2000
#   COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_astrophysical_jet_mach2000.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
#set_tests_labels(2D_ASTROPHYSICAL_JET_MACH2000 ASTROPHYSICAL_JET
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            EXTRA-LONG
#                                            INTEGRATION_TEST)