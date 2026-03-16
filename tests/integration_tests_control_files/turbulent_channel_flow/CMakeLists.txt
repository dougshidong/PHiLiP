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
configure_file(viscous_TCF_ILES_NSFR_cDG_IR-Roe_2PF_GLL_OI-0_Re395_p4_10x5x5_turbulent_initialization_not_cPlus_second_element_input.prm viscous_TCF_ILES_NSFR_cDG_IR-Roe_2PF_GLL_OI-0_Re395_p4_10x5x5_turbulent_initialization_not_cPlus_second_element_input.prm COPYONLY)
add_test(
  NAME VISCOUS_TURBULENT_CHANNEL_FLOW_QUANTITY_CHECK_cDG_IR-Roe_WALL_MODEL_C4_Re395_P4_10x5x5_SHORT
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TCF_ILES_NSFR_cDG_IR-Roe_2PF_GLL_OI-0_Re395_p4_10x5x5_turbulent_initialization_not_cPlus_second_element_input.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(VISCOUS_TURBULENT_CHANNEL_FLOW_QUANTITY_CHECK_cDG_IR-Roe_WALL_MODEL_C4_Re395_P4_10x5x5_SHORT 3D
                                                                                                              PARALLEL
                                                                                                              NAVIER_STOKES
                                                                                                              RUNGE-KUTTA
                                                                                                              STRONG
                                                                                                              COLLOCATED
                                                                                                              LES
                                                                                                              WALL_MODEL
                                                                                                              QUICK
                                                                                                              INTEGRATION_TEST)
# ----------------------------------------