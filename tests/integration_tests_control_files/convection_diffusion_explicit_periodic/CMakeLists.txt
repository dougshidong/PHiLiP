set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_conv_diff_explicit_periodic_energy.prm 2D_conv_diff_explicit_periodic_energy.prm COPYONLY)
add_test(
 NAME MPI_2D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_ENERGY_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_conv_diff_explicit_periodic_energy.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_ENERGY_LONG  CONVECTION_DIFFUSION_EXPLICIT_PERIODIC
                                                                            2D
                                                                            PARALLEL
                                                                            DIFFUSION
                                                                            RUNGE-KUTTA
                                                                            STRONG
                                                                            UNCOLLOCATED
                                                                            MODERATE
                                                                            INTEGRATION_TEST)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_conv_diff_explicit_periodic_OOA.prm 2D_conv_diff_explicit_periodic_OOA.prm COPYONLY)
add_test(
 NAME MPI_2D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_OOA_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_conv_diff_explicit_periodic_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_OOA_LONG CONVECTION_DIFFUSION_EXPLICIT_PERIODIC
                                                                        2D
                                                                        PARALLEL
                                                                        DIFFUSION
                                                                        RUNGE-KUTTA
                                                                        STRONG
                                                                        UNCOLLOCATED
                                                                        CONVERGENCE
                                                                        LONG
                                                                        INTEGRATION_TEST)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_conv_diff_explicit_periodic_energy.prm 1D_conv_diff_explicit_periodic_energy.prm COPYONLY)
add_test(
 NAME 1D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_ENERGY_LONG
COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_conv_diff_explicit_periodic_energy.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_ENERGY_LONG  CONVECTION_DIFFUSION_EXPLICIT_PERIODIC
                                                                        1D
                                                                        SERIAL
                                                                        DIFFUSION
                                                                        RUNGE-KUTTA
                                                                        STRONG
                                                                        UNCOLLOCATED
                                                                        QUICK
                                                                        INTEGRATION_TEST)
