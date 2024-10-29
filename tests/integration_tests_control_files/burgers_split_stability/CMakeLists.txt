set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_stability_energy.prm 1D_burgers_stability_energy.prm COPYONLY)
add_test(
  NAME 1D_BURGERS_STABILITY_ENERGY_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_stability_energy.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_STABILITY_ENERGY_LONG   BURGERS_SPLIT_STABILITY
                                                    1D
                                                    SERIAL
                                                    BURGERS_INVISCID
                                                    RUNGE-KUTTA
                                                    STRONG-SPLIT
                                                    UNCOLLOCATED
                                                    MODERATE
                                                    INTEGRATION_TEST)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_stability_OOA.prm 1D_burgers_stability_OOA.prm COPYONLY)
add_test(
  NAME 1D_BURGERS_STABILITY_ORDERS_OF_ACCURACY_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_stability_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_STABILITY_ORDERS_OF_ACCURACY_LONG   BURGERS_SPLIT_STABILITY
                                                                1D
                                                                SERIAL
                                                                BURGERS_INVISCID
                                                                RUNGE-KUTTA
                                                                STRONG-SPLIT
                                                                UNCOLLOCATED
                                                                CONVERGENCE
                                                                QUICK
                                                                INTEGRATION_TEST)
