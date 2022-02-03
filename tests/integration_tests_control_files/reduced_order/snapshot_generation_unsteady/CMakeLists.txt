set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(unsteady_snapshot_generator.sh unsteady_snapshot_generator.sh COPYONLY)
configure_file(1d_burgers_rewienski_snapshot_unsteady.prm 1d_burgers_rewienski_snapshot_unsteady.prm COPYONLY)
#add_test(
#        NAME 1D_BURGERS_REWIENSKI_SNAPSHOT_UNSTEADY
#        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_snapshot_unsteady.prm
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)