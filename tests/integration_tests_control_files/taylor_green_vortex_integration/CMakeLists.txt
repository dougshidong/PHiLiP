set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Viscous Taylor Green Vortex (TGV)
# =======================================
# ----------------------------------------
# -- Reference for flow setup: 
# -- -- Diosady, L., and S. Murman. "Case 3.3: Taylor green vortex evolution." Case Summary for 3rd International Workshop on Higher-Order CFD Methods. 2015.
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_weak_quick.prm viscous_taylor_green_vortex_energy_check_weak_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_WEAK_DG_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_weak_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_strong_quick.prm viscous_taylor_green_vortex_energy_check_strong_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_STRONG_DG_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_strong_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_weak_long.prm viscous_taylor_green_vortex_energy_check_weak_long.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_WEAK_DG_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_weak_long.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(remove_restart_files.sh remove_restart_files.sh COPYONLY)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_restart_check.prm viscous_taylor_green_vortex_restart_check.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_CHECK
  COMMAND bash -c 
  "./remove_restart_files.sh;
  numprocs=1 ;
  numprocstimestwo=$(( $numprocs * 2 )) ;
  while [[ $numprocstimestwo -le ${MPIMAX} ]];
  do 
    numprocs=$numprocstimestwo;
    numprocstimestwo=$(( $numprocs * 2 ));
  done ;
  mpirun -np $numprocs ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_restart_check.prm;
  return_val=$? ;
  if [ $return_val -ne 0 ]; then exit 1; else exit 0; fi"
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_restart_from_parameter_file_check.sh
viscous_taylor_green_vortex_restart_from_parameter_file_check.sh COPYONLY)
add_test(
  NAME MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_FROM_PARAMETER_FILE_CHECK
  COMMAND bash -c 
  "./viscous_taylor_green_vortex_restart_from_parameter_file_check.sh ${EXECUTABLE_OUTPUT_PATH}
  return_val1=$? ;
  numprocs=1 ;
  numprocstimestwo=$(( $numprocs * 2 )) ;
  while [[ $numprocstimestwo -le ${MPIMAX} ]];
  do 
    numprocs=$numprocstimestwo;
    numprocstimestwo=$(( $numprocs * 2 ));
  done ;
  mpirun -np $numprocs ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/restart-00004.prm;
  return_val2=$? ;
  if [ $return_val1 -ne 0 ] || [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_smagorinsky_model_energy_check_quick.prm viscous_TGV_LES_smagorinsky_model_energy_check_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_SMAGORINSKY_MODEL_ENERGY_CHECK_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_smagorinsky_model_energy_check_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_WALE_model_energy_check_quick.prm viscous_TGV_LES_WALE_model_energy_check_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_WALE_MODEL_ENERGY_CHECK_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_WALE_model_energy_check_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_vreman_model_energy_check_quick.prm viscous_TGV_LES_vreman_model_energy_check_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_VREMAN_MODEL_ENERGY_CHECK_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_vreman_model_energy_check_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_nsfr_gll_flux_nodes_quick.prm viscous_taylor_green_vortex_energy_check_nsfr_gll_flux_nodes_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_iLES_cDG_NSFR_GLL_FLUX_NODES_ENERGY_CHECK_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_nsfr_gll_flux_nodes_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_taylor_green_vortex_energy_check_nsfr_gl_flux_nodes_quick.prm viscous_taylor_green_vortex_energy_check_nsfr_gl_flux_nodes_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_iLES_cDG_NSFR_GL_FLUX_NODES_ENERGY_CHECK_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_taylor_green_vortex_energy_check_nsfr_gl_flux_nodes_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_energy_check_strong_adaptive_time_step.prm viscous_TGV_energy_check_strong_adaptive_time_step.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_STRONG_DG_ADAPTIVE_TIME_STEP
  COMMAND bash -c 
  "./remove_restart_files.sh;
  numprocs=1 ;
  numprocstimestwo=$(( $numprocs * 2 )) ;
  while [[ $numprocstimestwo -le ${MPIMAX} ]];
  do 
    numprocs=$numprocstimestwo;
    numprocstimestwo=$(( $numprocs * 2 ));
  done ;
  mpirun -np $numprocs ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_energy_check_strong_adaptive_time_step.prm;
  return_val=$? ;
  if [ $return_val -ne 0 ]; then exit 1; else exit 0; fi"
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_tgv_strong_dg_adaptive_time_step_from_restart.sh
viscous_tgv_strong_dg_adaptive_time_step_from_restart.sh COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_STRONG_DG_ADAPTIVE_TIME_STEP_FROM_RESTART
  COMMAND bash -c 
  "./viscous_tgv_strong_dg_adaptive_time_step_from_restart.sh ${EXECUTABLE_OUTPUT_PATH}
  return_val1=$? ;
  numprocs=1 ;
  numprocstimestwo=$(( $numprocs * 2 )) ;
  while [[ $numprocstimestwo -le ${MPIMAX} ]];
  do 
    numprocs=$numprocstimestwo;
    numprocstimestwo=$(( $numprocs * 2 ));
  done ;
  mpirun -np $numprocs ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/restart-00004.prm;
  return_val2=$? ;
  if [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_energy_check_strong_adaptive_time_step_output_restart_files_at_fixed_dt.prm viscous_TGV_energy_check_strong_adaptive_time_step_output_restart_files_at_fixed_dt.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_STRONG_DG_ADAPTIVE_TIME_STEP_OUTPUT_RESTART_FILES_AT_FIXED_DT
  COMMAND bash -c 
  "./remove_restart_files.sh;
  numprocs=1 ;
  numprocstimestwo=$(( $numprocs * 2 )) ;
  while [[ $numprocstimestwo -le ${MPIMAX} ]];
  do 
    numprocs=$numprocstimestwo;
    numprocstimestwo=$(( $numprocs * 2 ));
  done ;
  mpirun -np $numprocs ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_energy_check_strong_adaptive_time_step_output_restart_files_at_fixed_dt.prm;
  return_val=$? ;
  if [ $return_val -ne 0 ]; then exit 1; else exit 0; fi"
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_tgv_strong_dg_adaptive_time_step_from_restart_outputted_at_fixed_dt.sh
viscous_tgv_strong_dg_adaptive_time_step_from_restart_outputted_at_fixed_dt.sh COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_STRONG_DG_ADAPTIVE_TIME_STEP_FROM_RESTART_OUTPUTTED_AT_FIXED_DT
  COMMAND bash -c 
  "./viscous_tgv_strong_dg_adaptive_time_step_from_restart_outputted_at_fixed_dt.sh ${EXECUTABLE_OUTPUT_PATH}
  return_val1=$? ;
  numprocs=1 ;
  numprocstimestwo=$(( $numprocs * 2 )) ;
  while [[ $numprocstimestwo -le ${MPIMAX} ]];
  do 
    numprocs=$numprocstimestwo;
    numprocstimestwo=$(( $numprocs * 2 ));
  done ;
  mpirun -np $numprocs ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/restart-00002.prm;
  return_val2=$? ;
  if [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_smagorinsky_model_energy_check_nsfr_quick.prm viscous_TGV_LES_smagorinsky_model_energy_check_nsfr_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_SMAGORINSKY_MODEL_ENERGY_CHECK_NSFR_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_smagorinsky_model_energy_check_nsfr_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_shear_improved_smagorinsky_model_energy_check_nsfr_quick.prm viscous_TGV_LES_shear_improved_smagorinsky_model_energy_check_nsfr_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_SHEAR_IMPROVED_SMAGORINSKY_MODEL_ENERGY_CHECK_NSFR_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_shear_improved_smagorinsky_model_energy_check_nsfr_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_smagorinsky_model_energy_check_nsfr_legendre_proj_quick.prm viscous_TGV_LES_smagorinsky_model_energy_check_nsfr_legendre_proj_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_SMAGORINSKY_MODEL_ENERGY_CHECK_NSFR_LEGENDRE_PROJ_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_smagorinsky_model_energy_check_nsfr_legendre_proj_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
configure_file(viscous_TGV_LES_hpf_smagorinsky_model_energy_check_nsfr_quick.prm viscous_TGV_LES_hpf_smagorinsky_model_energy_check_nsfr_quick.prm COPYONLY)
add_test(
  NAME MPI_VISCOUS_TGV_LES_HPF_SMAGORINSKY_MODEL_ENERGY_CHECK_NSFR_QUICK
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/viscous_TGV_LES_hpf_smagorinsky_model_energy_check_nsfr_quick.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)