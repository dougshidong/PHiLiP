set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_burgers_inviscid_implicit.prm 1d_burgers_inviscid_implicit.prm COPYONLY)
add_test(
  NAME 1D_BURGERS_INVISCID_IMPLICIT_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_inviscid_implicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_INVISCID_IMPLICIT_MANUFACTURED_SOLUTION BURGERS_INVISCID_IMPLICIT
                                                                    1D
                                                                    SERIAL
                                                                    BURGERS_INVISCID
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    QUICK
                                                                    INTEGRATION_TEST)

configure_file(1d_burgers_inviscid_implicit_collocated.prm 1d_burgers_inviscid_implicit_collocated.prm COPYONLY)
add_test(
 NAME 1D_BURGERS_INVISCID_IMPLICIT_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_inviscid_implicit_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_INVISCID_IMPLICIT_COLLOCATED_MANUFACTURED_SOLUTION  BURGERS_INVISCID_IMPLICIT
                                                                                1D
                                                                                SERIAL
                                                                                BURGERS_INVISCID
                                                                                IMPLICIT
                                                                                WEAK
                                                                                UNCOLLOCATED
                                                                                QUICK
                                                                                INTEGRATION_TEST)

# configure_file(2d_burgers_inviscid_implicit.prm 2d_burgers_inviscid_implicit.prm COPYONLY)
# add_test(
#   NAME MPI_2D_BURGERS_INVISCID_IMPLICIT_MANUFACTURED_SOLUTION
#   COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_burgers_inviscid_implicit.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
# 
# configure_file(3d_burgers_inviscid_implicit.prm 3d_burgers_inviscid_implicit.prm COPYONLY)
# add_test(
#   NAME MPI_3D_BURGERS_INVISCID_IMPLICIT_MANUFACTURED_SOLUTION
#   COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_burgers_inviscid_implicit.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
