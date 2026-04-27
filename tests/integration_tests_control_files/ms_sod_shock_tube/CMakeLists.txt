#set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Multispecies Sod Shock Tube test
# =======================================
#if(${NUMBER_OF_SPECIES} EQUAL 2)
#configure_file(1D_ms_sod_shock_tube.prm 1D_ms_sod_shock_tube.prm COPYONLY)
#add_test(
#  NAME 1D_MULTISPECIES_SOD_SHOCK_TUBE_TEST
#  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_ms_sod_shock_tube.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(1D_MULTISPECIES_SOD_SHOCK_TUBE_TEST 	MS_SOD_SHOCK_TUBE
#														1D
#														SERIAL
#														MULTISPECIES
#														RUNGE-KUTTA
#														COLLOCATED
#														LIMITER
#														QUICK
#														INTEGRATION_TEST)
#endif()
