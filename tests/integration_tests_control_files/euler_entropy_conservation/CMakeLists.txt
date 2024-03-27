set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Entropy conservation test 
# =======================================
# ----------------------------------------
# - Runs a short time on TGV
# - test will fail if entropy is not conserved
# - Checks all split fluxes
# ----------------------------------------
configure_file(euler_entropy_conserving_split_forms_check.prm euler_entropy_conserving_split_forms_check.prm COPYONLY)
add_test(
    NAME MPI_3D_EULER_ENTROPY_CONSERVING_SPLIT_FORMS_CHECK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/euler_entropy_conserving_split_forms_check.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------

# =======================================
# Entropy conservation test 
# =======================================
# ----------------------------------------
# - Runs full time on inviscid TGV
# - Included for test set-up; should not be included in ctest
# ----------------------------------------
#configure_file(inviscid_tgv_entropy_check_long.prm inviscid_tgv_entropy_check_long.prm COPYONLY)
#add_test(
#    NAME MPI_3D_EULER_RRK_ENTROPY_CONSERVING_CHECK_LONG
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_tgv_entropy_check_long.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

# ----------------------------------------
# =======================================
# Entropy conservation test 
# =======================================
# ----------------------------------------
# - Runs full time on viscous TGV
# - Included for test set-up; should not be included in ctest
# ----------------------------------------
#configure_file(viscous_tgv_entropy_check_long.prm viscous_tgv_entropy_check_long.prm COPYONLY)
#add_test(
#    NAME MPI_3D_NAVIER_STOKES_RRK_ENTROPY_CHECK_LONG
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_tgv_entropy_check_long.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
