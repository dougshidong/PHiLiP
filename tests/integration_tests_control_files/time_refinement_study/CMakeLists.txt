set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

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
