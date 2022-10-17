set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Inviscid Taylor Green Vortex (TGV)
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
#configure_file(inviscid_taylor_green_vortex.prm inviscid_taylor_green_vortex.prm COPYONLY)
#add_test(
#  NAME MPI_INVISCID_TAYLOR_GREEN_VORTEX
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_taylor_green_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
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

# =======================================
# Viscous Taylor Green Vortex (TGV)
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex.prm viscous_taylor_green_vortex.prm COPYONLY)
#add_test(
#  NAME SERIAL_VISCOUS_TAYLOR_GREEN_VORTEX
#  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex.prm viscous_taylor_green_vortex.prm COPYONLY)
#add_test(
#  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex_large_eddy_simulation.prm viscous_taylor_green_vortex_large_eddy_simulation.prm COPYONLY)
#add_test(
#  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_LARGE_EDDY_SIMULATION
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_large_eddy_simulation.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
#configure_file(viscous_taylor_green_vortex_large_eddy_simulation.prm viscous_taylor_green_vortex_large_eddy_simulation.prm COPYONLY)
#add_test(
#  NAME SERIAL_VISCOUS_TAYLOR_GREEN_VORTEX_LARGE_EDDY_SIMULATION
#  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_large_eddy_simulation.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# ----------------------------------------
