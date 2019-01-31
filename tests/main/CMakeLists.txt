set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})#/test_output)
add_test(
  NAME Main
  COMMAND PHiLiP
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}#/linear_advection
)
