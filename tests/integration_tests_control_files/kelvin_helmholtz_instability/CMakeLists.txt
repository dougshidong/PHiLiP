set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Kelvin Helmholtz Instability -- 2D 
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(2D_kelvin_helmoltz_instability.prm 2D_kelvin_helmoltz_instability.prm COPYONLY)
add_test(
  NAME MPI_2D_KELVIN_HELMHOLTZ_INSTABILITY
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_kelvin_helmoltz_instability.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
