set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(2d_euler_gaussian_bump.prm 2d_euler_gaussian_bump.prm COPYONLY)
add_test(
  NAME 2D_EULER_GAUSSIAN_BUMP
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
