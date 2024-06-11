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

# =======================================
# Vortex advection 1D
# =======================================
# ----------------------------------------
# Vortex advectiion 1D (Air composition)
# ----------------------------------------
configure_file(1d_multi_species_vortex_advection.prm 1d_multi_species_vortex_advection.prm COPYONLY)
add_test(
  NAME MPI_1D_MULTI_SPECIES_VORTEX_ADVECTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advectiion 1D (Euler)
# ----------------------------------------
configure_file(1d_euler_vortex_advection.prm 1d_euler_vortex_advection.prm COPYONLY)
add_test(
  NAME MPI_1D_EULER_VORTEX_ADVECTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_vortex_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advectiion 1D (Euler, error test)
# ----------------------------------------
configure_file(1d_euler_vortex_advection_error_study.prm 1d_euler_vortex_advection_error_study.prm COPYONLY)
add_test(
  NAME MPI_1D_EULER_VORTEX_ERROR_STUDY
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_vortex_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Vortex advectiion 1D (Euler, error test)
# ----------------------------------------
configure_file(1d_multi_species_vortex_advection_error_study.prm 1d_multi_species_vortex_advection_error_study.prm COPYONLY)
add_test(
  NAME MPI_1D_MULTI_SPECIES_VORTEX_ERROR_STUDY
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Bubble advectiion 1D (Euler, error test)
# ----------------------------------------
configure_file(1d_euler_bubble_advection_error_study.prm 1d_euler_bubble_advection_error_study.prm COPYONLY)
add_test(
  NAME MPI_1D_EULER_BUBBLE_ERROR_STUDY
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_bubble_advection_error_study.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# Isentropic Euler Vortex 2D (Euler)
# ----------------------------------------
configure_file(2d_multi_species_isentropic_euler_vortex.prm 2d_multi_species_isentropic_euler_vortex.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_ISENTROPIC_EULER_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_isentropic_euler_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)