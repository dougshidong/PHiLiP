#set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
#configure_file(1D_advection_limiter.prm 1D_advection_limiter.prm COPYONLY)
#add_test(
# NAME 1D_Advection_Limiter_Test
# COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_advection_limiter.prm
# WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(1D_Advection_Limiter_Test  ADVECTION_LIMITER
#                                            1D
#                                            SERIAL
#                                            ADVECTION
#                                            RUNGE-KUTTA
#                                            STRONG-SPLIT
#                                            COLLOCATED
#                                            LIMITER
#                                            QUICK
#                                            INTEGRATION_TEST)

configure_file(1D_advection_limiter_OOA.prm 1D_advection_limiter_OOA.prm COPYONLY)
add_test(
 NAME 1D_Advection_Limiter_OOA_Test
 COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_advection_limiter_OOA.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set_tests_labels(1D_Advection_Limiter_OOA_Test  ADVECTION_LIMITER
                                            1D
                                            SERIAL
                                            ADVECTION
                                            RUNGE-KUTTA
                                            STRONG
                                            COLLOCATED
                                            LIMITER
                                            CONVERGENCE
                                            QUICK
                                            INTEGRATION_TEST)

configure_file(1D_advection_limiter_OOA_NSFR.prm 1D_advection_limiter_OOA_NSFR.prm COPYONLY)
add_test(
 NAME 1D_Advection_Limiter_OOA_NSFR_Test
 COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_advection_limiter_OOA_NSFR.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set_tests_labels(1D_Advection_Limiter_OOA_NSFR_Test  ADVECTION_LIMITER
                                            1D
                                            SERIAL
                                            ADVECTION
                                            RUNGE-KUTTA
                                            STRONG-SPLIT
                                            COLLOCATED
                                            LIMITER
                                            CONVERGENCE
                                            QUICK
                                            INTEGRATION_TEST)

#configure_file(2D_advection_limiter.prm 2D_advection_limiter.prm COPYONLY)
#add_test(
# NAME 2D_Advection_Limiter_Test
# COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_limiter.prm
# WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(2D_Advection_Limiter_Test  ADVECTION_LIMITER
#                                            2D
#                                            SERIAL
#                                            ADVECTION
#                                            RUNGE-KUTTA
#                                            STRONG
#                                            COLLOCATED
#                                            LIMITER
#                                            QUICK
#                                            INTEGRATION_TEST)
                                            
configure_file(2D_advection_limiter_OOA.prm 2D_advection_limiter_OOA.prm COPYONLY)
add_test(
 NAME 2D_Advection_Limiter_OOA_Test
 COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_limiter_OOA.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_Advection_Limiter_OOA_Test  ADVECTION_LIMITER
                                            2D
                                            PARALLEL
                                            ADVECTION
                                            RUNGE-KUTTA
                                            STRONG
                                            COLLOCATED
                                            LIMITER
                                            CONVERGENCE
                                            MODERATE
                                            INTEGRATION_TEST)

configure_file(2D_advection_limiter_OOA_NSFR.prm 2D_advection_limiter_OOA_NSFR.prm COPYONLY)
add_test(
 NAME 2D_Advection_Limiter_OOA_NSFR_Test
 COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_limiter_OOA_NSFR.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(2D_Advection_Limiter_OOA_NSFR_Test  ADVECTION_LIMITER
                                            2D
                                            PARALLEL
                                            ADVECTION
                                            RUNGE-KUTTA
                                            STRONG-SPLIT
                                            COLLOCATED
                                            LIMITER
                                            CONVERGENCE
                                            MODERATE
                                            INTEGRATION_TEST)