set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_advection_implicit.prm 1d_advection_implicit.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION    ADVECTION_IMPLICIT
                                                                1D
                                                                SERIAL
                                                                ADVECTION
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                MANUFACTURED_SOLUTION
                                                                QUICK
                                                                INTEGRATION_TEST)

configure_file(2d_advection_implicit.prm 2d_advection_implicit.prm COPYONLY)
add_test(
  NAME 2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION    ADVECTION_IMPLICIT
                                                                2D
                                                                SERIAL
                                                                ADVECTION
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                MANUFACTURED_SOLUTION
                                                                MODERATE
                                                                INTEGRATION_TEST)

configure_file(2d_advection_explicit.prm 2d_advection_explicit.prm COPYONLY)
add_test(
  NAME MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_explicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_LONG   ADVECTION_IMPLICIT
                                                                        2D
                                                                        PARALLEL
                                                                        ADVECTION
                                                                        RUNGE-KUTTA
                                                                        WEAK
                                                                        UNCOLLOCATED
                                                                        MANUFACTURED_SOLUTION
                                                                        LONG
                                                                        INTEGRATION_TEST)

configure_file(2d_advection_implicit.prm 2d_advection_implicit.prm COPYONLY)
add_test(
  NAME MPI_2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION    ADVECTION_IMPLICIT
                                                                    2D
                                                                    PARALLEL
                                                                    ADVECTION
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(3d_advection_implicit.prm 3d_advection_implicit.prm COPYONLY)
add_test(
  NAME MPI_3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION    ADVECTION_IMPLICIT
                                                                    3D
                                                                    PARALLEL
                                                                    ADVECTION
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    MODERATE
                                                                    INTEGRATION_TEST)

# Tests for strong form
configure_file(1d_advection_implicit_strong.prm 1d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG ADVECTION_IMPLICIT
                                                                    1D
                                                                    SERIAL
                                                                    ADVECTION
                                                                    IMPLICIT
                                                                    STRONG
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    EXPECTED_FAILURE
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(2d_advection_implicit_strong.prm 2d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME 2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG ADVECTION_IMPLICIT
                                                                    2D
                                                                    SERIAL
                                                                    ADVECTION
                                                                    IMPLICIT
                                                                    STRONG
                                                                    UNCOLLOCATED
                                                                    MANUFACTURED_SOLUTION
                                                                    EXPECTED_FAILURE
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(3d_advection_implicit_strong.prm 3d_advection_implicit_strong.prm COPYONLY)
add_test(
  NAME MPI_3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG_MEDIUM
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_implicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_3D_ADVECTION_IMPLICIT_MANUFACTURED_SOLUTION_STRONG_MEDIUM  ADVECTION_IMPLICIT
                                                                                3D
                                                                                PARALLEL
                                                                                ADVECTION
                                                                                IMPLICIT
                                                                                STRONG
                                                                                UNCOLLOCATED
                                                                                MANUFACTURED_SOLUTION
                                                                                EXPECTED_FAILURE
                                                                                QUICK
                                                                                INTEGRATION_TEST)

configure_file(1d_advection_implicit_collocated.prm 1d_advection_implicit_collocated.prm COPYONLY)
add_test(
 NAME 1D_ADVECTION_IMPLICIT_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_implicit_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_ADVECTION_IMPLICIT_COLLOCATED_MANUFACTURED_SOLUTION ADVECTION_IMPLICIT
                                                                        1D
                                                                        SERIAL
                                                                        ADVECTION
                                                                        IMPLICIT
                                                                        WEAK
                                                                        COLLOCATED
                                                                        MANUFACTURED_SOLUTION
                                                                        QUICK
                                                                        INTEGRATION_TEST)
configure_file(1d_advection_explicit_strong.prm 1d_advection_explicit_strong.prm COPYONLY)
add_test(
  NAME 1D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_LONG_STRONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_advection_explicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_LONG_STRONG    ADVECTION_IMPLICIT
                                                                            1D
                                                                            SERIAL
                                                                            ADVECTION
                                                                            RUNGE-KUTTA
                                                                            STRONG
                                                                            UNCOLLOCATED
                                                                            MANUFACTURED_SOLUTION
                                                                            MODERATE
                                                                            INTEGRATION_TEST)

configure_file(2d_advection_explicit_strong.prm 2d_advection_explicit_strong.prm COPYONLY)
add_test(
  NAME MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_STRONG_LONG
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_advection_explicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_STRONG_LONG    ADVECTION_IMPLICIT
                                                                                2D
                                                                                PARALLEL
                                                                                ADVECTION
                                                                                RUNGE-KUTTA
                                                                                STRONG
                                                                                UNCOLLOCATED
                                                                                MANUFACTURED_SOLUTION
                                                                                MODERATE
                                                                                INTEGRATION_TEST)

configure_file(3d_advection_explicit_strong.prm 3d_advection_explicit_strong.prm COPYONLY)
add_test(
  NAME MPI_3D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_STRONG
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_advection_explicit_strong.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_3D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION_STRONG ADVECTION_IMPLICIT
                                                                        3D
                                                                        PARALLEL
                                                                        ADVECTION
                                                                        RUNGE-KUTTA
                                                                        STRONG
                                                                        UNCOLLOCATED
                                                                        MANUFACTURED_SOLUTION
                                                                        LONG
                                                                        INTEGRATION_TEST)
