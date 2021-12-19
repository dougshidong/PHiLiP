set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_conv_diff_explicit_periodic_energy.prm 2D_conv_diff_explicit_periodic_energy.prm COPYONLY)
add_test(
 NAME MPI_2D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_ENERGY_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_conv_diff_explicit_periodic_energy.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_conv_diff_explicit_periodic_OOA.prm 2D_conv_diff_explicit_periodic_OOA.prm COPYONLY)
add_test(
 NAME MPI_2D_CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_OOA_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_conv_diff_explicit_periodic_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
