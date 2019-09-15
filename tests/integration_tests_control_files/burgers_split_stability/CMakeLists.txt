set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_stability.prm 1D_burgers_stability.prm COPYONLY)
add_test(
  NAME 1D_burgers_stability_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_stability.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
