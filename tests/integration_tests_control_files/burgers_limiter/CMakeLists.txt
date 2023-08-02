set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_limiter.prm 1D_burgers_limiter.prm COPYONLY)
add_test(
  NAME 1D_Burgers_Limiter_Test
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_limiter.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1D_burgers_limiter_OOA.prm 1D_burgers_limiter_OOA.prm COPYONLY)
add_test(
  NAME 1D_Burgers_Limiter_OOA_Test
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_limiter_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1D_burgers_limiter_OOA_NSFR.prm 1D_burgers_limiter_OOA_NSFR.prm COPYONLY)
add_test(
  NAME 1D_Burgers_Limiter_OOA_NSFR_Test
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_limiter_OOA_NSFR.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2D_burgers_limiter.prm 2D_burgers_limiter.prm COPYONLY)
add_test(
   NAME 2D_Burgers_Limiter_Test
   COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_burgers_limiter.prm
   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2D_burgers_limiter_OOA.prm 2D_burgers_limiter_OOA.prm COPYONLY)
add_test(
  NAME 2D_Burgers_Limiter_OOA_Test
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_burgers_limiter_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2D_burgers_limiter_OOA_NSFR.prm 2D_burgers_limiter_OOA_NSFR.prm COPYONLY)
add_test(
  NAME 2D_Burgers_Limiter_OOA_NSFR_Test
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_burgers_limiter_OOA_NSFR.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
