add_subdirectory(setup_files)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# ===================================================
# Decaying Homogeneous Isotropic Turbulence (DHIT)
# ===================================================
# ----------------------------------------
# -- Reference for flow setup [TO BE UPDATED]: 
# -- -- Diosady, L., and S. Murman. "Case 3.3: Taylor green vortex evolution." Case Summary for 3rd International Workshop on Higher-Order CFD Methods. 2015.
# ----------------------------------------
configure_file(dhit_init_check_serial.prm dhit_init_check_serial.prm COPYONLY)
add_test(NAME SERIAL_DHIT_INIT_CHECK
         COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/dhit_init_check_serial.prm
         WORKING_DIRECTORY ${TEST_OUTPUT_DIR})
# ----------------------------------------
configure_file(dhit_init_check_mpi.prm dhit_init_check_mpi.prm COPYONLY)
add_test(NAME MPI_DHIT_INIT_CHECK
         COMMAND mpirun -np 4 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/dhit_init_check_mpi.prm
         WORKING_DIRECTORY ${TEST_OUTPUT_DIR})
# ----------------------------------------
