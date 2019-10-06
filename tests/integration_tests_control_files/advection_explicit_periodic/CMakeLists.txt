set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_advection_explicit_periodic.prm 2D_advection_explicit_periodic.prm COPYONLY)
add_test(
	NAME MPI_2D_ADVECTION_EXPLICIT_PERIODIC_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_explicit_periodic.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
