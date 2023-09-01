set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Inviscid Real Gas Tests
# =======================================
# ----------------------------------------
# add details here
# ----------------------------------------
configure_file(2d_gaussian_bump_inviscid_air.prm 2d_gaussian_bump_inviscid_air.prm COPYONLY)
add_test(
  NAME MPI_2D_GAUSSIAN_BUMP_INVISCID_AIR
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gaussian_bump_inviscid_air.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# add details here
# ----------------------------------------
configure_file(2d_gaussian_bump_inviscid_nitrogen.prm 2d_gaussian_bump_inviscid_nitrogen.prm COPYONLY)
add_test(
  NAME MPI_2D_GAUSSIAN_BUMP_INVISCID_NITROGEN
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gaussian_bump_inviscid_nitrogen.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
