set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_nonsmooth_case.prm 2D_nonsmooth_case.prm COPYONLY)
add_test(
 NAME 2D_NONSMOOTH_CASE
 COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_nonsmooth_case.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

