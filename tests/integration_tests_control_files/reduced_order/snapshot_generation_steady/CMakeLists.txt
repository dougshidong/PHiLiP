set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(steady_state_snapshot_generator.sh steady_state_snapshot_generator.sh COPYONLY)
configure_file(1d_burgers_rewienski_snapshot_steady.prm 1d_burgers_rewienski_snapshot_steady.prm COPYONLY)
#add_test(
#        NAME 1D_BURGERS_REWIENSKI_SNAPSHOT_STEADY
#        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_snapshot_steady.prm
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)