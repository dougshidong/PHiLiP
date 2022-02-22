set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(3D_euler_split_inviscid_taylor_green_vortex.prm 3D_euler_split_inviscid_taylor_green_vortex.prm COPYONLY)
add_test(
    NAME MPI_3D_EULER_SPLIT_TAYLOR_GREEN_LONG
    COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_euler_split_inviscid_taylor_green_vortex.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
