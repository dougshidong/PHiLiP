set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Entropy conservation test 
# =======================================
# ----------------------------------------
# - Runs a short time on TGV
# - test will fail if entropy is not conserved
# ----------------------------------------
configure_file(euler_ismail_roe_entropy_check.prm euler_ismail_roe_entropy_check.prm COPYONLY)
add_test(
    NAME MPI_3D_EULER_ISMAIL_ROE_ENTROPY_CHECK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/euler_ismail_roe_entropy_check.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
