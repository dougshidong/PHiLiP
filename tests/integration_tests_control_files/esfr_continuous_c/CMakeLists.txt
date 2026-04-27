# =======================================
# Advection -- 1D -- h refinement  -- ESFR parameter range
# =======================================
# ----------------------------------------
# - Parameters are  chosen to run in a reasonable amount of time
# - Number of refinements and end time should be increased for better results
# ----------------------------------------

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_advection_explicit_periodic_OOA_continuous_c_NSFR.prm 1D_advection_explicit_periodic_OOA_continuous_c_NSFR.prm COPYONLY)
add_test(
 NAME 1D_ADVECTION_EXPLICIT_PERIODIC_OOA_CONTINUOUS_C_NSFR_LONG
COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_advection_explicit_periodic_OOA_continuous_c_NSFR.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_ADVECTION_EXPLICIT_PERIODIC_OOA_CONTINUOUS_C_NSFR_LONG  	    ESFR_CONTINUOUS_C
                                                                                            1D
                                                                                            REFINEMENT_STUDY
                                                                                            SERIAL
                                                                                            ADVECTION
                                                                                            RUNGE-KUTTA
                                                                                            STRONG-SPLIT
                                                                                            COLLOCATED
                                                                                            CONVERGENCE
                                                                                            MODERATE
                                                                                            INTEGRATION_TEST)
# =======================================
# Burgers -- 1D -- h refinement  -- ESFR parameter range
# =======================================
# ----------------------------------------
# - Parameters are  chosen to run in a reasonable amount of time
# - Number of refinements and end time should be increased for better results
# ----------------------------------------
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_OOA_ESFR_parameter_range.prm 1D_burgers_OOA_ESFR_parameter_range.prm COPYONLY)
add_test(
  NAME 1D_BURGERS_OOA_ESFR_RANGE_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_OOA_ESFR_parameter_range.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_OOA_ESFR_RANGE_LONG  					    ESFR_CONTINUOUS_C
                                                                                            1D
                                                                                            SERIAL
                                                                                            BURGERS_INVISCID
                                                                                            REFINEMENT_STUDY
                                                                                            RUNGE-KUTTA
                                                                                            STRONG-SPLIT
                                                                                            COLLOCATED
                                                                                            CONVERGENCE
                                                                                            QUICK
                                                                                            INTEGRATION_TEST)
# =======================================
# Inviscid Isentropic Vortex -- 2D -- h refinement  -- ESFR parameter range
# =======================================
# ----------------------------------------
# - Parameters are  chosen to run in a reasonable amount of time
# - Number of refinements and end time should be increased for better results
# ----------------------------------------
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_inviscid_isentropic_vortex_OOA_ESFR_parameter_range.prm 2D_inviscid_isentropic_vortex_OOA_ESFR_parameter_range.prm COPYONLY)
add_test(
  NAME MPI_2D_H_REFINEMENT_ESFR_RANGE_INVISCID_ISENTROPIC_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex_OOA_ESFR_parameter_range.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(MPI_2D_H_REFINEMENT_ESFR_RANGE_INVISCID_ISENTROPIC_VORTEX ESFR_CONTINUOUS_C
                                                                	   2D
                                                                           PARALLEL
                                                                           REFINEMENT_STUDY
                                                                           EULER
                                                                           RUNGE-KUTTA
                                                                           STRONG-SPLIT
                                                                           COLLOCATED
                                                                           CONVERGENCE
                                                                           LONG
                                                                           INTEGRATION_TEST)
