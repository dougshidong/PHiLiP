set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
# ----------------------------------------
configure_file(skin_friction_computation_check.prm skin_friction_computation_check.prm COPYONLY)
add_test(
  NAME TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK_FOR_FRICTION_REYNOLDS_NUMBER_395
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/skin_friction_computation_check.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK_FOR_FRICTION_REYNOLDS_NUMBER_395 3D
                                                                                        PARALLEL
                                                                                        NAVIER_STOKES
                                                                                        RUNGE-KUTTA
                                                                                        WEAK
                                                                                        COLLOCATED
                                                                                        QUICK
                                                                                        INTEGRATION_TEST)
# ----------------------------------------
configure_file(skin_friction_computation_check_Re5200.prm skin_friction_computation_check_Re5200.prm COPYONLY)
add_test(
  NAME TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK_FOR_FRICTION_REYNOLDS_NUMBER_5200
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/skin_friction_computation_check_Re5200.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK_FOR_FRICTION_REYNOLDS_NUMBER_5200 3D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            RUNGE-KUTTA
                                                                            WEAK
                                                                            COLLOCATED
                                                                            QUICK
                                                                            INTEGRATION_TEST)
# ----------------------------------------
configure_file(skin_friction_computation_check_Re5200_wall_model.prm skin_friction_computation_check_Re5200_wall_model.prm COPYONLY)
add_test(
  NAME TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK_FOR_FRICTION_REYNOLDS_NUMBER_5200_WALL_MODEL
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/skin_friction_computation_check_Re5200_wall_model.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK_FOR_FRICTION_REYNOLDS_NUMBER_5200_WALL_MODEL 3D
                                                                            PARALLEL
                                                                            NAVIER_STOKES
                                                                            RUNGE-KUTTA
                                                                            WEAK
                                                                            COLLOCATED
                                                                            LES
                                                                            QUICK
                                                                            INTEGRATION_TEST)
# ----------------------------------------