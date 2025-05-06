set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(2d_optimization_inverse_manufactured.prm 2d_optimization_inverse_manufactured.prm COPYONLY)
add_test(
  NAME 2D_OPTIMIZATION_INVERSE_MANUFACTURED
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_optimization_inverse_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_OPTIMIZATION_INVERSE_MANUFACTURED   OPTIMIZATION_INVERSE_MANUFACTURED
                                                        2D
                                                        PARALLEL
                                                        EULER
                                                        IMPLICIT
                                                        WEAK
                                                        UNCOLLOCATED
                                                        OPTIMIZATION
                                                        EXPECTED_FAILURE
                                                        MODERATE
                                                        OPTIMIZATION
                                                        INTEGRATION_TEST)
