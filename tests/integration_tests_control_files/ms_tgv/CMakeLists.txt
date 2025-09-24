set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Taylor Green Vortex 3D
# =======================================
# ----------------------------------------
# Taylor Green Vortex 3D (Multi-Species, Uniform mass fractions)
# ----------------------------------------
configure_file(3d_multi_species_taylor_green_vortex.prm 3d_multi_species_taylor_green_vortex.prm COPYONLY)
add_test(
  NAME MPI_3D_MULTI_SPECIES_TAYLOR_GREEN_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/3d_multi_species_taylor_green_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

 set_tests_labels(MPI_3D_MULTI_SPECIES_TAYLOR_GREEN_VORTEX  MS_TGV
 							      3D
 							      PARALLEL
 							      REAL-GAS
 							      MULTI-SPECIES
 							      RUNGE-KUTTA
 							      WEAK
 							      UNCOLLOCATED
 							      EXTRA-LONG
 							      INTEGRATION_TEST)
# ----------------------------------------
# Taylor Green Vortex 3D (Multi-Species, Mixture mass fractions)
# ----------------------------------------
configure_file(3d_multi_species_mixture_taylor_green_vortex.prm 3d_multi_species_mixture_taylor_green_vortex.prm COPYONLY)
add_test(
  NAME MPI_3D_MULTI_SPECIES_MIXTURE_TAYLOR_GREEN_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/3d_multi_species_mixture_taylor_green_vortex.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

 set_tests_labels(MPI_3D_MULTI_SPECIES_MIXTURE_TAYLOR_GREEN_VORTEX  MS_TGV
                                                                    3D
                                                                    PARALLEL
                                                                    REAL-GAS
                                                                    MULTI-SPECIES
                                                                    RUNGE-KUTTA
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    EXTRA-LONG
                                                                    INTEGRATION_TEST)
