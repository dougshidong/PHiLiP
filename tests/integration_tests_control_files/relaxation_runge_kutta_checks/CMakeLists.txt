# =======================================
# Energy conservation test for inviscid Burgers 
# =======================================
# ----------------------------------------
# Four calculations are performed: (small dt, large dt) x (RRK, no RRK)
# Initial condition is sinusoidal; boundaries are periodic
# Energy is expected to be conserved in all cases except (large dt, no RRK)
# Test will fail if energy is not conserved in the other three cases
# ----------------------------------------

configure_file(1D_burgers_energy_conservation_rrk_cDG.prm 1D_burgers_energy_conservation_rrk_cDG.prm COPYONLY)
add_test(
    NAME 1D_BURGERS_ENERGY_CONSERVATION_RRK_CDG
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_energy_conservation_rrk_cDG.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# OOA for inviscid Burgers
# =======================================
# ----------------------------------------
# Time step refinement study for RRK
# ----------------------------------------

configure_file(1D_burgers_rrk_temporal_OOA_cDG.prm 1D_burgers_rrk_temporal_OOA_cDG.prm COPYONLY)
add_test(
    NAME 1D_BURGERS_RRK_TEMPORAL_OOA_CDG
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_rrk_temporal_OOA_cDG.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Numerical entropy conservation test for Euler
# =======================================
# ----------------------------------------
# Four calculations are performed: (small dt, large dt) x (RRK, no RRK)
# Initial condition: TGV 
# Energy is expected to be conserved in all cases except (large dt, no RRK)
# Test will fail if energy is not conserved in the other three cases
# ----------------------------------------

configure_file(3D_euler_numerical_entropy_conservation_rrk_cDG.prm 3D_euler_numerical_entropy_conservation_rrk_cDG.prm COPYONLY)
add_test(
    NAME MPI_3D_EULER_NUMERICAL_ENTROPY_CONSERVATION_RRK_CDG
    COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_euler_numerical_entropy_conservation_rrk_cDG.prm
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

configure_file(1D_burgers_energy_conservation_rrk_cPlus.prm 1D_burgers_energy_conservation_rrk_cPlus.prm COPYONLY)
add_test(
    NAME 1D_BURGERS_ENERGY_CONSERVATION_RRK_CPLUS
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_energy_conservation_rrk_cPlus.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# OOA for inviscid Burgers
# =======================================
# ----------------------------------------
# Time step refinement study for RRK
# ----------------------------------------

configure_file(1D_burgers_rrk_temporal_OOA_cPlus.prm 1D_burgers_rrk_temporal_OOA_cPlus.prm COPYONLY)
add_test(
    NAME 1D_BURGERS_RRK_TEMPORAL_OOA_CPLUS
    COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_rrk_temporal_OOA_cPlus.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Numerical entropy conservation test for Euler
# =======================================
# ----------------------------------------
# Four calculations are performed: (small dt, large dt) x (RRK, no RRK)
# Initial condition: TGV 
# Energy is expected to be conserved in all cases except (large dt, no RRK)
# Test will fail if energy is not conserved in the other three cases
# ----------------------------------------

configure_file(3D_euler_numerical_entropy_conservation_rrk_cPlus.prm 3D_euler_numerical_entropy_conservation_rrk_cPlus.prm COPYONLY)
add_test(
    NAME MPI_3D_EULER_NUMERICAL_ENTROPY_CONSERVATION_RRK_CPLUS
    COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_euler_numerical_entropy_conservation_rrk_cPlus.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Check that NS does not abort for cPlus RRK
# =======================================
# ----------------------------------------
# Do a short viscous TGV test using cPlus RRK to ensure there is no segfault etc.
# This test will not have a pass/fail condition as we don't know an exact entropy change rate
# ----------------------------------------

configure_file(3D_navier_stokes_RRK_check.prm 3D_navier_stokes_RRK_check.prm COPYONLY)
add_test(
    NAME MPI_3D_VISCOUS_TGV_RRK_CHECK_QUICK
    COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_navier_stokes_RRK_check.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# =======================================                                                                                                                                                                          
# Inviscid Isentropic Vortex -- 2D -- h refinement
# =======================================
# ----------------------------------------
# - Check that orders are correct when using RRK
# - Parameters are  chosen to run in a reasonable amount of time
# - Number of refinements and end time should be increased for better results
# ----------------------------------------
configure_file(2D_inviscid_isentropic_vortex_h_refinement.prm 2D_inviscid_isentropic_vortex_h_refinement.prm COPYONLY)
add_test(
    NAME MPI_2D_H_REFINEMENT_RRK_INVISCID_ISENTROPIC_VORTEX
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_inviscid_isentropic_vortex_h_refinement.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

