set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Riemann Problem
# =======================================

configure_file(1D_riemann_problem.prm 1D_riemann_problem.prm COPYONLY)
add_test(
  NAME 1D_RIEMANN_PROBLEM
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_riemann_problem.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_RIEMANN_PROBLEM 1D_RIEMANN
                                    1D
                                    SERIAL
                                    EULER
                                    RUNGE-KUTTA
                                    STRONG-SPLIT
                                    COLLOCATED
                                    LIMITER
                                    MODERATE
                                    INTEGRATION_TEST)