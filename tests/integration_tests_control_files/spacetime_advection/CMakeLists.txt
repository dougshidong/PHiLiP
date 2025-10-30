# TO DO BEFORE MERGING: add comments + labels

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1Dp1_linear_advection.prm 1Dp1_linear_advection.prm COPYONLY)
add_test(
    NAME serial_1Dp1_ADVECTION
COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/1Dp1_linear_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1Dp1_linear_advection_c_ramp.prm 1Dp1_linear_advection_c_ramp.prm COPYONLY)
add_test(
    NAME serial_1Dp1_ADVECTION_C_RAMP
COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/1Dp1_linear_advection_c_ramp.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)
