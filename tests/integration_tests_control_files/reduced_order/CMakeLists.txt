set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_burgers_reduced_order.prm 1d_burgers_reduced_order.prm COPYONLY)

add_test(
        NAME 1D_BURGERS_REDUCED_ORDER
        COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_reduced_order.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_subdirectory(snapshot_generation)
