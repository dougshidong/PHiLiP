set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_stability_energy.prm 1D_burgers_stability_energy.prm COPYONLY)
add_test(
  NAME 1D_BURGERS_STABILITY_ENERGY_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_stability_energy.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_stability_OOA.prm 1D_burgers_stability_OOA.prm COPYONLY)
add_test(
  NAME 1D_BURGERS_STABILITY_ORDERS_OF_ACCURACY_LONG
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_stability_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


# =======================================
# Energy conservation test for inviscid Burgers 
# =======================================
# ----------------------------------------
# Four calculations are performed: (small dt, large dt) x (RRK, no RRK)
# Initial condition is sinusoidal; boundaries are periodic
# Energy is expected to be conserved in all cases except (large dt, no RRK)
# Test will fail if energy is not conserved in the other three cases
# ----------------------------------------

configure_file(1D_burgers_energy_conservation_rrk.prm 1D_burgers_energy_conservation_rrk.prm COPYONLY)
add_test(
    NAME 1D_BURGERS_ENERGY_CONSERVATION_RRK
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_energy_conservation_rrk.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
