set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d-burgers-inviscid-explicit.prm 1d-burgers-inviscid-explicit.prm COPYONLY)
add_test(
	NAME 1D_BURGERS_INVISCID_EXPLICIT_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d-burgers-inviscid-explicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
