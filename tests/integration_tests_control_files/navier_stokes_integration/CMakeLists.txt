set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Manufactured solutions
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_navier_stokes_l2roe_manufactured.prm 1d_navier_stokes_l2roe_manufactured.prm COPYONLY)
add_test(
  NAME 1D_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_navier_stokes_l2roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured_navah.prm 2d_euler_laxfriedrichs_manufactured_navah.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_NAVIER_STOKES_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured_navah.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_roe_manufactured_navah.prm 2d_euler_roe_manufactured_navah.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_NAVIER_STOKES_ROE_MANUFACTURED_SOLUTION_NAVAH_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_roe_manufactured_navah.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_l2roe_manufactured_navah.prm 2d_euler_l2roe_manufactured_navah.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_NAVIER_STOKES_L2ROE_MANUFACTURED_SOLUTION_NAVAH_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_l2roe_manufactured_navah.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
