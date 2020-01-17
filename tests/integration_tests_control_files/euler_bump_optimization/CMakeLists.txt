set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(2d_euler_bump_optimization.prm 2d_euler_bump_optimization.prm COPYONLY)
add_test(
  NAME 2D_EULER_BUMP_OPTIMIZATION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_bump_optimization.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
