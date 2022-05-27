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

# =======================================
# POD Adaptive sampling
# =======================================
configure_file(1d_burgers_rewienski_adaptive_sampling.prm 1d_burgers_rewienski_adaptive_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_ADAPTIVE_SAMPLING
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_adaptive_sampling.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
# POD Adaptive sampling testing
# =======================================
configure_file(1d_burgers_rewienski_adaptive_sampling_testing.prm 1d_burgers_rewienski_adaptive_sampling_testing.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_ADAPTIVE_SAMPLING_TESTING
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_adaptive_sampling_testing.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(burgers_rewienski_snapshot_generator.sh burgers_rewienski_snapshot_generator.sh COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_GENERATE_SNAPSHOTS
        COMMAND bash -c
        "./burgers_rewienski_snapshot_generator.sh ${EXECUTABLE_OUTPUT_PATH}"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(inviscid_transonic_naca0012_adaptive_sampling.prm inviscid_transonic_naca0012_adaptive_sampling.prm COPYONLY)
configure_file(naca0012_hopw_ref2.msh naca0012_hopw_ref2.msh COPYONLY)
add_test(
        NAME INVISCID_TRANSONIC_NACA0012_ADAPTIVE_SAMPLING
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_transonic_naca0012_adaptive_sampling.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(naca0012_snapshot_generator.sh naca0012_snapshot_generator.sh COPYONLY)
add_test(
        NAME NACA0012_GENERATE_SNAPSHOTS
        COMMAND bash -c
        "./naca0012_snapshot_generator.sh ${EXECUTABLE_OUTPUT_PATH}"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(naca0012_reduced_order.prm naca0012_reduced_order.prm COPYONLY)
add_test(
        NAME NACA0012_REDUCED_ORDER
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/naca0012_reduced_order.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(naca0012_reduced_order_testing.prm naca0012_reduced_order_testing.prm COPYONLY)
add_test(
        NAME NACA0012_REDUCED_ORDER_TESTING
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/naca0012_reduced_order_testing.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(burgers_rewienski_reduced_order.prm burgers_rewienski_reduced_order.prm COPYONLY)
add_test(
        NAME BURGERS_REWIENSKI_REDUCED_ORDER
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/burgers_rewienski_reduced_order.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

#configure_file(1d_burgers_rewienski_testing.prm 1d_burgers_rewienski_testing.prm COPYONLY)
#add_test(
#        NAME 1D_BURGERS_REWIENSKI_TESTING
#        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_testing.prm;
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

configure_file(1d_burgers_rewienski_fd_sensitivity.prm 1d_burgers_rewienski_fd_sensitivity.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_SENSITIVITY
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_fd_sensitivity.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

#configure_file(1d_burgers_viscous_fd_sensitivity.prm 1d_burgers_viscous_fd_sensitivity.prm COPYONLY)
#add_test(
#        NAME 1D_BURGERS_VISCOUS_SENSITIVITY
#        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_viscous_fd_sensitivity.prm
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