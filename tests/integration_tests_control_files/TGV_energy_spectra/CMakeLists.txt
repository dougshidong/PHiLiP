add_subdirectory(solution_files)
add_subdirectory(restart_files)
add_subdirectory(flow_field_files)
##Linear grid cDG
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_energy_spectra.prm 3D_TGV_energy_spectra.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TGV_ENERGY_SPECTRA                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_energy_spectra.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(parameters_for_assembling_mpi_files.txt parameters_for_assembling_mpi_files.txt COPYONLY)                                                                                                                                            
