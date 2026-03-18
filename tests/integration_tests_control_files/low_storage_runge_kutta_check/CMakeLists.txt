set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Time Study (Linear Advection Explicit RK)
# =======================================
# ----------------------------------------
# Time refinement study on linear advection using a sinusoidal initial condition
# L2 error calculated with respect to the exact solution
# Test will fail if the convergence order is not close to the expected order
# ----------------------------------------
configure_file(low_storage_rk_test.prm low_storage_rk_test.prm COPYONLY)
add_test(
    NAME 1D_ADVECTION_LSRK
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/low_storage_rk_test.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_ADVECTION_LSRK  LOW_STORAGE_RUNGE-KUTTA_CHECK
                                    1D
                                    REFINEMENT_STUDY
                                    SERIAL
                                    ADVECTION
                                    LOW_STORAGE_RUNGE-KUTTA
                                    WEAK
                                    UNCOLLOCATED
                                    CONVERGENCE
                                    QUICK
                                    INTEGRATION_TEST)
# ----------------------------------------


configure_file(taylor_green_vortex_error_adaptive.prm taylor_green_vortex_error_adaptive.prm COPYONLY)
add_test(
    NAME MPI_3D_LSRK_ERROR_ADAPTIVE_TGV
    COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/taylor_green_vortex_error_adaptive.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_3D_LSRK_ERROR_ADAPTIVE_TGV     LOW_STORAGE_RUNGE-KUTTA_CHECK
                                                    3D
                                                    PARALLEL
                                                    EULER
                                                    LOW_STORAGE_RUNGE-KUTTA
                                                    STRONG-SPLIT
                                                    UNCOLLOCATED
                                                    QUICK
                                                    INTEGRATION_TEST)
