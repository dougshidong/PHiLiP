set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Double Mach Reflection Case
# =======================================
# configure_file(2D_double_mach_reflection.prm 2D_double_mach_reflection.prm COPYONLY)
# add_test(
#   NAME 2D_DOUBLE_MACH_REFLECTION
#   COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_double_mach_reflection.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
#set_tests_labels(2D_DOUBLE_MACH_REFLECTION DOUBLE_MACH_REFLECTION
#                                            2D
#                                            PARALLEL
#                                            EULER
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            EXTRA-LONG
#                                            INTEGRATION_TEST)