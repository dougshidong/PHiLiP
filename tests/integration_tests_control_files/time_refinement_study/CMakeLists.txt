set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Time Study (Linear Advection Explicit RK)
# =======================================
# ----------------------------------------
# Time refinement study on linear advection using a sinusoidal initial condition
# L2 error calculated with respect to the exact solution
# Test will fail if the convergence order is not close to the expected order
# ----------------------------------------
configure_file(time_refinement_study_advection_explicit.prm time_refinement_study_advection_explicit.prm COPYONLY)
add_test(
    NAME 1D_TIME_REFINEMENT_STUDY_ADVECTION_EXPLICIT
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/time_refinement_study_advection_explicit.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------

# =======================================
# Time Study (Inviscid Burgers Explicit RK)
# =======================================
# ----------------------------------------
# Time refinement study on inviscid Burgers using a sinusoidal initial condition
# L2 error calculated with respect to a reference solution with a very small timestep at the actual end time
# RRK can be used by changing the ODE solver in the .prm file
# Test will fail if the convergence order is not close to the expected order
# ----------------------------------------
# ----------------------------------------
configure_file(time_refinement_study_burgers_explicit.prm time_refinement_study_burgers_explicit.prm COPYONLY)
add_test(
    NAME 1D_TIME_REFINEMENT_STUDY_BURGERS_EXPLICIT
    COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/time_refinement_study_burgers_explicit.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
