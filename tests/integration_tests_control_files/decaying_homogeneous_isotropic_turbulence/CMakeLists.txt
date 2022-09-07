set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# ===================================================
# Decaying Homogeneous Isotropic Turbulence (DHIT)
# ===================================================
# ----------------------------------------
# -- Reference for flow setup: 
# -- -- Diosady, L., and S. Murman. "Case 3.3: Taylor green vortex evolution." Case Summary for 3rd International Workshop on Higher-Order CFD Methods. 2015.
# ----------------------------------------
configure_file(dhit_init_check.prm dhit_init_check.prm COPYONLY)
add_test(NAME MPI_DHIT_INIT_CHECK
         COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/dhit_init_check.prm
         WORKING_DIRECTORY ${TEST_OUTPUT_DIR})
# ----------------------------------------
