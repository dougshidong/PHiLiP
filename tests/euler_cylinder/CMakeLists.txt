set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(2d_euler_cylinder.prm 2d_euler_cylinder.prm COPYONLY)
add_test(
  NAME 2D_EULER_CYLINDER
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_cylinder.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
