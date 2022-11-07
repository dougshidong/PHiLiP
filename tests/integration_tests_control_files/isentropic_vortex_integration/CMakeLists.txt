set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Inviscid Isentropic Vortex -- 2D
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(2D_inviscid_isentropic_vortex.prm 2D_inviscid_isentropic_vortex.prm COPYONLY)
add_test(
  NAME SERIAL_2D_INVISCID_ISENTROPIC_VORTEX
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------

# =======================================
# Inviscid Isentropic Vortex -- 2D
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(2D_inviscid_isentropic_vortex.prm 2D_inviscid_isentropic_vortex.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_ISENTROPIC_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------

# =======================================
# Inviscid Isentropic Vortex -- 2D -- time refinement
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(2D_inviscid_isentropic_vortex_time_refinement.prm 2D_inviscid_isentropic_vortex_time_refinement.prm COPYONLY)
add_test(
  NAME MPI_2D_INVISCID_ISENTROPIC_VORTEX_TIME_REFINEMENT
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex_time_refinement.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# =======================================
# Inviscid Isentropic Vortex -- 3D
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(3D_inviscid_isentropic_vortex.prm 3D_inviscid_isentropic_vortex.prm COPYONLY)
add_test(
  NAME MPI_3D_INVISCID_ISENTROPIC_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_inviscid_isentropic_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
