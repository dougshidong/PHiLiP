##Linear grid cDG
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_split_form.prm 3D_TGV_scaling_split_form.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_SPLIT_FORM                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_split_form.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_overint.prm 3D_TGV_scaling_overint.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_OVERINTEGRATION                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_overint.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_cons.prm 3D_TGV_scaling_cons.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_CONSERVATIVE                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_cons.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       

##Linear grid cHU
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_split_form_cHU.prm 3D_TGV_scaling_split_form_cHU.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_SPLIT_FORM_CHU                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_split_form_cHU.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_overint_cHU.prm 3D_TGV_scaling_overint_cHU.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_OVERINTEGRATION_CHU                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_overint_cHU.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_cons_cHU.prm 3D_TGV_scaling_cons_cHU.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_CONSERVATIVE_CHU                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_cons_cHU.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       

##Curvilinear grid DG

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_split_form_curv.prm 3D_TGV_scaling_split_form_curv.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_SPLIT_FORM_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_split_form_curv.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_overint_curv.prm 3D_TGV_scaling_overint_curv.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_OVERINTEGRATION_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_overint_curv.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_cons_curv.prm 3D_TGV_scaling_cons_curv.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_CONSERVATIVE_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_cons_curv.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       

##Curvilinear Grid cHU
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_split_form_curv_cHU.prm 3D_TGV_scaling_split_form_curv_cHU.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_SPLIT_FORM_CHU_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_split_form_curv_cHU.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_overint_curv_cHU.prm 3D_TGV_scaling_overint_curv_cHU.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_OVERINTEGRATION_CHU_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_overint_curv_cHU.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_TGV_scaling_cons_curv_cHU.prm 3D_TGV_scaling_cons_curv_cHU.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_TAYLOR_GREEN_SCALING_CONSERVATIVE_CHU_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_TGV_scaling_cons_curv_cHU.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
