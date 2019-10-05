set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Manufactured solutions
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_euler_laxfriedrichs_manufactured.prm 1d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_euler_laxfriedrichs_manufactured_collocated.prm 1d_euler_laxfriedrichs_manufactured_collocated.prm COPYONLY)
add_test(
	NAME 1D_EULER_LAXFRIEDRICHS_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_laxfriedrichs_manufactured_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


configure_file(2d_euler_laxfriedrichs_manufactured.prm 2d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME 2D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured_collocated.prm 2d_euler_laxfriedrichs_manufactured_collocated.prm COPYONLY)
add_test(
	NAME 2D_EULER_LAXFRIEDRICHS_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


configure_file(2d_euler_laxfriedrichs_manufactured.prm 2d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_euler_roe_manufactured.prm 1d_euler_roe_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_ROE_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_roe_manufactured.prm 2d_euler_roe_manufactured.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_ROE_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Exact solutions

configure_file(2d_euler_cylinder.prm 2d_euler_cylinder.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_CYLINDER_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_cylinder.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_gaussian_bump.prm 2d_euler_gaussian_bump.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Vortex test case takes wayyy too much time. It works, so uncomment below if you want to wait.
# configure_file(2d_euler_vortex.prm 2d_euler_vortex.prm COPYONLY)
# add_test(
#   NAME 2D_EULER_INTEGRATION_VORTEX
#   mpirun -n 1 COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_vortex.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )


#configure_file(2d_euler_entropy_waves.prm 2d_euler_entropy_waves.prm COPYONLY)
#add_test(
#  NAME 2D_EULER_INTEGRATION_ENTROPY_WAVES
#  mpirun -n 1 COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_entropy_waves.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

#configure_file(3d_euler_laxfriedrichs_manufactured.prm 3d_euler_laxfriedrichs_manufactured.prm COPYONLY)
#add_test(
#  NAME 3D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
#  mpirun -n 1 COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_euler_laxfriedrichs_manufactured.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
