set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Sod Shock Tube test
# =======================================

configure_file(2D_sedov_blast_wave.prm 2D_sedov_blast_wave.prm COPYONLY)
add_test(
  NAME 2D_SEDOV_BLAST_WAVE
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_sedov_blast_wave.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
