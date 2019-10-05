set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_advection_implicit.prm 1d_advection_implicit.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_advection_implicit.prm 2d_advection_implicit.prm COPYONLY)
add_test(
  NAME 2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_advection_explicit.prm 2d_advection_explicit.prm COPYONLY)
add_test(
  NAME MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_explicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_advection_implicit.prm 2d_advection_implicit.prm COPYONLY)
add_test(
  NAME MPI_2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(3d_advection_implicit.prm 3d_advection_implicit.prm COPYONLY)
add_test(
  NAME MPI_3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Tests for strong form
configure_file(1d_advection_implicit_strong.prm 1d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_advection_implicit_strong.prm 2d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(3d_advection_implicit_strong.prm 3d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME MPI_3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG_MEDIUM
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_advection_implicit_collocated.prm 1d_advection_implicit_collocated.prm COPYONLY)
add_test(
	NAME 1D_ADVECTION_IMPLICIT_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

