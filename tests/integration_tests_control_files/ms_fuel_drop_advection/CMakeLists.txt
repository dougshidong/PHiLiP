set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Fuel Drop Advection 2D
# =======================================
# ----------------------------------------
# Isentropic Drop Advection (Multi-Species)
# ----------------------------------------
configure_file(2d_multi_species_fuel_drop_advection.prm 2d_multi_species_fuel_drop_advection.prm COPYONLY)
add_test(
  NAME MPI_2D_MULTI_SPECIES_FUEL_DROP_ADVECTION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D_2species -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_fuel_drop_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
