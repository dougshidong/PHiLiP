set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_advection_implicit.prm 1d_advection_implicit.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_advection_implicit.prm 2d_advection_implicit.prm COPYONLY)
add_test(
  NAME 2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(3d_advection_implicit.prm 3d_advection_implicit.prm COPYONLY)
add_test(
  NAME 3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Tests for strong form
configure_file(1d_advection_implicit_strong.prm 1d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_advection_implicit_strong.prm 2d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(3d_advection_implicit_strong.prm 3d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
