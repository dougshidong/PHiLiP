set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_advection_limiter.prm 1D_advection_limiter.prm COPYONLY)
add_test(
 NAME 1D_Advection_Limiter_Test
 COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_advection_limiter.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2D_advection_limiter.prm 2D_advection_limiter.prm COPYONLY)
add_test(
 NAME 2D_Advection_Limiter_Test
 COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_limiter.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
