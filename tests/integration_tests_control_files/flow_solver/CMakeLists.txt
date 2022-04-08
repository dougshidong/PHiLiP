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

# =======================================
# Time Study (Linear Advection Explicit RK)
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(time_study_advection_explicit.prm time_study_advection_explicit.prm COPYONLY)
add_test(
  NAME 1D_Time_Study_Advection_Explicit
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/time_study_advection_explicit.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
