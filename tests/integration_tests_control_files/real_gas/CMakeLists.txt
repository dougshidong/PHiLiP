set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Inviscid Real Gas Tests
# =======================================

# =======================================
# Gaussian Bump 2D
# =======================================
# ----------------------------------------
# Gaussian Bump 2D (Air)
# ----------------------------------------
configure_file(2d_gaussian_bump_inviscid_air.prm 2d_gaussian_bump_inviscid_air.prm COPYONLY)
add_test(
  NAME MPI_2D_GAUSSIAN_BUMP_INVISCID_AIR
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gaussian_bump_inviscid_air.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Gaussian Bump 2D (Nitrogen)
# ----------------------------------------
configure_file(2d_gaussian_bump_inviscid_nitrogen.prm 2d_gaussian_bump_inviscid_nitrogen.prm COPYONLY)
add_test(
  NAME MPI_2D_GAUSSIAN_BUMP_INVISCID_NITROGEN
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_gaussian_bump_inviscid_nitrogen.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Accoustic Wave 2D
# =======================================
# ----------------------------------------
# Accoustic Wave 2D (Air)
# ----------------------------------------
configure_file(2d_inviscid_acoustic_wave_air.prm 2d_inviscid_acoustic_wave_air.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_ACOUSTIC_WAVE_AIR
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_inviscid_acoustic_wave_air.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Accoustic Wave 2D (Nitrogen)
# ----------------------------------------
configure_file(2d_inviscid_acoustic_wave_nitrogen.prm 2d_inviscid_acoustic_wave_nitrogen.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_ACOUSTIC_WAVE_NITROGEN
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_inviscid_acoustic_wave_nitrogen.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# primitive to conservative variable check
# ----------------------------------------
configure_file(2d_primitive_to_conservative_monospecies_real_gas_vs_euler.prm 2d_primitive_to_conservative_monospecies_real_gas_vs_euler.prm COPYONLY)
add_test(
  NAME MPI_2D_REAL_GAS_PRIMITIVE_TO_CONSERVATIVE_VS_EULER
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_primitive_to_conservative_monospecies_real_gas_vs_euler.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
