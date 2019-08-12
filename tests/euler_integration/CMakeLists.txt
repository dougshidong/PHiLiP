set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(2d_euler_vortex.prm 2d_euler_vortex.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_VORTEX
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_cylinder.prm 2d_euler_cylinder.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_CYLINDER
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_cylinder.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_entropy_waves.prm 2d_euler_entropy_waves.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_ENTROPY_WAVES
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_entropy_waves.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_gaussian_bump.prm 2d_euler_gaussian_bump.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_GAUSSIAN_BUMP
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Manufactured solutions
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_euler_laxfriedrichs_manufactured.prm 1d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured.prm 2d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME 2D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_euler_roe_manufactured.prm 1d_euler_roe_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_ROE_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_roe_manufactured.prm 2d_euler_roe_manufactured.prm COPYONLY)
add_test(
  NAME 2D_EULER_ROE_MANUFACTURED_SOLUTION
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

#configure_file(3d_euler_laxfriedrichs_manufactured.prm 3d_euler_laxfriedrichs_manufactured.prm COPYONLY)
#add_test(
#  NAME 3D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
#  COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_euler_laxfriedrichs_manufactured.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
