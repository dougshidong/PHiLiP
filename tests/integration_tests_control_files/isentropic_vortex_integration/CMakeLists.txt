set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Inviscid Isentropic Vortex -- 2D
# =======================================
# ----------------------------------------
# - Commented; could be used for debugging.
# ----------------------------------------
#configure_file(2D_inviscid_isentropic_vortex.prm 2D_inviscid_isentropic_vortex.prm COPYONLY)
#add_test(
#  NAME SERIAL_2D_INVISCID_ISENTROPIC_VORTEX
#  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------

# =======================================
# Inviscid Isentropic Vortex -- 2D
# =======================================
# ----------------------------------------
# - Commented; convenient for outputting visualizations
# ----------------------------------------
#configure_file(2D_inviscid_isentropic_vortex.prm 2D_inviscid_isentropic_vortex.prm COPYONLY)
#add_test(
#  NAME MPI_2D_INVISCID_ISENTROPIC_VORTEX
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------

# =======================================
# Inviscid Isentropic Vortex -- 2D -- h refinement
# =======================================
# ----------------------------------------
# - Parameters are  chosen to run in a reasonable amount of time
# - Number of refinements and end time should be increased for better results
# ----------------------------------------
configure_file(2D_inviscid_isentropic_vortex_h_refinement.prm 2D_inviscid_isentropic_vortex_h_refinement.prm COPYONLY)
add_test(
  NAME MPI_2D_H_REFINEMENT_INVISCID_ISENTROPIC_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex_h_refinement.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
# =======================================
# Inviscid Isentropic Vortex -- 3D
# =======================================
# ----------------------------------------
# - Commented out as it's very long and not tested yet. 
# ----------------------------------------
#configure_file(3D_inviscid_isentropic_vortex.prm 3D_inviscid_isentropic_vortex.prm COPYONLY)
#add_test(
#  NAME MPI_3D_INVISCID_ISENTROPIC_VORTEX
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_inviscid_isentropic_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
