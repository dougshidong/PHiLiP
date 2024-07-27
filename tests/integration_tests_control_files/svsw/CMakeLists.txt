set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Sod Shock Tube test
# =======================================
configure_file(2D_svsw.prm 2D_svsw.prm COPYONLY)
add_test(
  NAME 2D_SVSW_TEST
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_svsw.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


