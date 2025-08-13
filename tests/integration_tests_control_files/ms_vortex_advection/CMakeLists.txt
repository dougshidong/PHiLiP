set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# === Multi Species Tests ===
# =======================================

# =======================================
# Vortex advection 1D
# =======================================
# ----------------------------------------
# Vortex advection 1D (Air composition)
# ----------------------------------------
configure_file(1d_multi_species_vortex_advection.prm 1d_multi_species_vortex_advection.prm COPYONLY)
add_test(
  NAME 1D_MULTI_SPECIES_VORTEX_ADVECTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# ----------------------------------------
# Vortex advection 1D (Multi-Species, error test)
# ----------------------------------------
configure_file(1d_multi_species_vortex_advection_error_study.prm 1d_multi_species_vortex_advection_error_study.prm COPYONLY)
add_test(
  NAME 1D_MULTI_SPECIES_VORTEX_ERROR_STUDY
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advection 1D (Multi-Species_High-Temperature, error test)
# ----------------------------------------
configure_file(1d_multi_species_high_temperature_vortex_advection_error_study.prm 1d_multi_species_high_temperature_vortex_advection_error_study.prm COPYONLY)
add_test(
  NAME 1D_MULTI_SPECIES_HIGH_TEMPERATURE_VORTEX_ERROR_STUDY
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_high_temperature_vortex_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advection 1D (Multi-Species, Calorically perfect)
# ----------------------------------------
configure_file(1d_multi_species_calorically_perfect_euler_vortex_advection.prm 1d_multi_species_calorically_perfect_euler_vortex_advection.prm COPYONLY)
add_test(
  NAME 1D_MULTI_SPECIES_CALORICALLY_PERFECT_VORTEX_ADVECTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_calorically_perfect_euler_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advection 1D (Multi-Species, Calorically perfect, error test)
# ----------------------------------------
configure_file(1d_multi_species_calorically_perfect_euler_vortex_advection_error_study.prm 1d_multi_species_calorically_perfect_euler_vortex_advection_error_study.prm COPYONLY)
add_test(
  NAME 1D_MULTI_SPECIES_CALORICALLY_PERFECT_VORTEX_ERROR_STUDY
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_calorically_perfect_euler_vortex_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Vortex advection 2D
# =======================================
# ----------------------------------------
# Vortex advection 2D (Multi-Species)
# ----------------------------------------
configure_file(2d_multi_species_vortex_advection.prm 2d_multi_species_vortex_advection.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_VORTEX_ADVECTION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advection 2D (Multi-Species, error test)
# ----------------------------------------
configure_file(2d_multi_species_vortex_advection_error_study.prm 2d_multi_species_vortex_advection_error_study.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_VORTEX_ERROR_STUDY
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_vortex_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Isentropic Euler Vortex 2D
# =======================================
# ----------------------------------------
# Isentropic Euler Vortex 2D (Euler)
# ----------------------------------------
configure_file(2d_multi_species_isentropic_euler_vortex.prm 2d_multi_species_isentropic_euler_vortex.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_ISENTROPIC_EULER_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_isentropic_euler_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Vortex advection 3D
# =======================================
# ----------------------------------------
# Vortex advectiion 3D (Multi-Species)
# ----------------------------------------
configure_file(3d_multi_species_vortex_advection.prm 3d_multi_species_vortex_advection.prm COPYONLY)
add_test(
  NAME MPI_3D_MULTI_SPECIES_VORTEX_ADVECTION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/3d_multi_species_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
