set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 2D Multispecies Isentropic Vortex test
# =======================================
#if(${NUMBER_OF_SPECIES} EQUAL 2)
#configure_file(2D_multi_species_isentropic_vortex.prm 2D_multi_species_isentropic_vortex.prm COPYONLY)
#add_test(
#  NAME 2D_MULTISPECIES_ISENTROPIC_VORTEX_TEST
#  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_multi_species_isentropic_vortex.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
#set_tests_labels(2D_MULTISPECIES_ISENTROPIC_VORTEX_TEST MS_ISENTROPIC_VORTEX_TEST
#														2D
#														PARALLEL
#														MULTISPECIES
#														RUNGE-KUTTA
#														COLLOCATED
#														LIMITER
#														QUICK
#														INTEGRATION_TEST)
#endif()
