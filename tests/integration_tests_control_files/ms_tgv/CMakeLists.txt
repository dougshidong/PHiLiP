set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_real_gas_split_inviscid_taylor_green_vortex.prm 3D_real_gas_split_inviscid_taylor_green_vortex.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME 3D_REAL_GAS_SPLIT_TAYLOR_GREEN_KEP_TEST                                                                                                                                                                                  
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_real_gas_split_inviscid_taylor_green_vortex.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)
set_tests_labels(3D_REAL_GAS_SPLIT_TAYLOR_GREEN_KEP_TEST  REAL_GAS_SPLIT_TAYLOR_GREEN_KEP_TEST
                                                          3D
                                                          PARALLEL
                                                          REAL-GAS
                                                          RUNGE-KUTTA
                                                          STRONG-SPLIT
                                                          UNCOLLOCATED
                                                          EXTRA-LONG
                                                          INTEGRATION_TEST)