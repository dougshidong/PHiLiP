set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# === Multi Species Tests ===
# =======================================
# ----------------------------------------
# Vortex advectiion 1D (MS-CP-Euler)
# ----------------------------------------
configure_file(1d_multi_species_calorically_perfect_euler_vortex_advection.prm 1d_multi_species_calorically_perfect_euler_vortex_advection.prm COPYONLY)
add_test(
  NAME MPI_1D_MULTI_SPECIES_CALORICALLY_PERFECT_EULER_VORTEX_ADVECTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_calorically_perfect_euler_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)