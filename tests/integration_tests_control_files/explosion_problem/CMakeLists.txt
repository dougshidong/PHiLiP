set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D/3D Explosion Problem
# =======================================

configure_file(2D_explosion_problem.prm 2D_explosion_problem.prm COPYONLY)
add_test(
  NAME 2D_EXPLOSION_PROBLEM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_explosion_problem.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(3D_explosion_problem.prm 3D_explosion_problem.prm COPYONLY)
add_test(
  NAME 3D_EXPLOSION_PROBLEM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_explosion_problem.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
