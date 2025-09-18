set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# === Multi Species Tests ===
# =======================================

# =======================================
# Accoustic Wave 2D
# =======================================
# ----------------------------------------
# Accoustic Wave 2D (Air composition)
# ----------------------------------------
configure_file(2d_multi_species_acoustic_wave.prm 2d_multi_species_acoustic_wave.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_ACOUSTIC_WAVE
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_acoustic_wave.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

 set_tests_labels(MPI_2D_MULTI_SPECIES_ACOUSTIC_WAVE    MS_ACOUSTIC_WAVE
                                                        2D
                                                        PARALLEL
                                                        REAL-GAS
                                                        MULTI-SPECIES
                                                        RUNGE-KUTTA
                                                        WEAK
                                                        UNCOLLOCATED
                                                        QUICK
                                                        INTEGRATION_TEST)
