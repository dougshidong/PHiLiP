set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(snapshot_generation_unsteady)
add_subdirectory(snapshot_generation_steady)


# =======================================
# Test Galerkin and Petrov-Galerkin reduced-order solvers
# =======================================
#Be careful when using bash in cmake, the test will pass if the only last command executed successfully (i.e. if the last command is rm *.txt, the test will always pass)
configure_file(1d_burgers_rewienski_reduced_order.prm 1d_burgers_rewienski_reduced_order.prm COPYONLY)
configure_file(1d_burgers_rewienski_snapshot_unsteady.prm 1d_burgers_rewienski_snapshot_unsteady.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_REDUCED_ORDER
        COMMAND bash -c
        "rm *.txt ;
        mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_snapshot_unsteady.prm;
        return_val1=$? ;
        mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_reduced_order.prm ;
        return_val2=$? ;
        rm *.txt ;
        if [ $return_val1 -ne 0 ] || [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# Test POD adaptation
# =======================================
configure_file(1d_burgers_rewienski_pod_adaptation.prm 1d_burgers_rewienski_pod_adaptation.prm COPYONLY)
configure_file(pod_adaptation_snapshot_generator.sh pod_adaptation_snapshot_generator.sh COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_POD_ADAPTATION
        COMMAND bash -c
        "rm *.txt ;
        ./pod_adaptation_snapshot_generator.sh ${EXECUTABLE_OUTPUT_PATH};
        mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_pod_adaptation.prm;
        return_val=$? ;
        rm *.txt ;
        if [ $return_val -ne 0 ]; then exit 1; else exit 0; fi"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

#add_test(
#        NAME 1D_BURGERS_REWIENSKI_TESTING
#        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_pod_adaptation.prm;
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

# =======================================
# Default manufactured solution
# =======================================
configure_file(1d_burgers_rewienski_manufactured.prm 1d_burgers_rewienski_manufactured.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_MANUFACTURED_SOLUTION
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_manufactured.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)