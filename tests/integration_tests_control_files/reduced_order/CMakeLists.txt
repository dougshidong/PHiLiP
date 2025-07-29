set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Test Burgers Rewienski Galerkin and Petrov-Galerkin reduced-order solvers
# =======================================
#Be careful when using bash in cmake, the test will pass if the only last command executed successfully (i.e. if the last command is rm *.txt, the test will always pass)
configure_file(1d_burgers_rewienski_reduced_order_consistency_snapshots.sh 1d_burgers_rewienski_reduced_order_consistency_snapshots.sh COPYONLY)
configure_file(1d_burgers_rewienski_reduced_order_consistency.prm 1d_burgers_rewienski_reduced_order_consistency.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_REDUCED_ORDER_CONSISTENCY
        COMMAND bash -c
        "rm *.txt ;
        ./1d_burgers_rewienski_reduced_order_consistency_snapshots.sh ${EXECUTABLE_OUTPUT_PATH}
        return_val1=$? ;
        mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_reduced_order_consistency.prm ;
        return_val2=$? ;
        rm *.txt ;
        if [ $return_val1 -ne 0 ] || [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_REWIENSKI_REDUCED_ORDER_CONSISTENCY REDUCED_ORDER
                                                                1D
                                                                SERIAL
                                                                BURGERS_REWIENSKI
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                MODERATE
                                                                INTEGRATION_TEST)
# =======================================
# Burgers Rewienski Adaptive Sampling (Two Parameters)
# =======================================
configure_file(1d_burgers_rewienski_adaptive_sampling.prm 1d_burgers_rewienski_adaptive_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_ADAPTIVE_SAMPLING
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_adaptive_sampling.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_REWIENSKI_ADAPTIVE_SAMPLING REDUCED_ORDER
                                                        1D
                                                        SERIAL
                                                        BURGERS_REWIENSKI
                                                        IMPLICIT
                                                        WEAK
                                                        UNCOLLOCATED
                                                        LONG
                                                        INTEGRATION_TEST)

# =======================================
# Burgers Rewienski Adaptive Sampling (One Parameter - generates results for hyperreduction)
# =======================================
configure_file(1d_burgers_rewienski_one_param_adaptive_sampling.prm 1d_burgers_rewienski_one_param_adaptive_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_ONE_PARAM_ADAPTIVE_SAMPLING
        COMMAND bash -c
        "mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_one_param_adaptive_sampling.prm
        mkdir -p 1d_burgers_one_param ;
        mv rom_table_iteration_final.txt ./1d_burgers_one_param ;
        mv snapshot_table_iteration_final.txt ./1d_burgers_one_param ;
        mv solution_snapshots_iteration_final.txt ./1d_burgers_one_param ; "
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_REWIENSKI_ONE_PARAM_ADAPTIVE_SAMPLING   REDUCED_ORDER
                                                                    1D
                                                                    SERIAL
                                                                    BURGERS_REWIENSKI
                                                                    IMPLICIT
                                                                    WEAK
                                                                    UNCOLLOCATED
                                                                    MODERATE
                                                                    INTEGRATION_TEST)

# =======================================
# Burgers Rewienski Adaptive Sampling Testing
# =======================================
#configure_file(1d_burgers_rewienski_adaptive_sampling_testing.prm 1d_burgers_rewienski_adaptive_sampling_testing.prm COPYONLY)
#add_test(
#        NAME 1D_BURGERS_REWIENSKI_ADAPTIVE_SAMPLING_TESTING
#        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_adaptive_sampling_testing.prm
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

# =======================================
# Burgers Rewienski Snapshot Generator
# =======================================
#configure_file(burgers_rewienski_snapshot_generator.sh burgers_rewienski_snapshot_generator.sh COPYONLY)
#add_test(
#        NAME 1D_BURGERS_REWIENSKI_GENERATE_SNAPSHOTS
#        COMMAND bash -c
#        "./burgers_rewienski_snapshot_generator.sh ${EXECUTABLE_OUTPUT_PATH}"
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

# =======================================
# Test Inviscid NACA0012 Galerkin and Petrov-Galerkin reduced-order solvers
# =======================================
#Be careful when using bash in cmake, the test will pass if the only last command executed successfully (i.e. if the last command is rm *.txt, the test will always pass)
configure_file(inviscid_naca0012_reduced_order_consistency_snapshots.sh inviscid_naca0012_reduced_order_consistency_snapshots.sh COPYONLY)
configure_file(inviscid_naca0012_reduced_order_consistency.prm inviscid_naca0012_reduced_order_consistency.prm COPYONLY)
add_test(
        NAME INVISCID_NACA0012_REDUCED_ORDER_CONSISTENCY
        COMMAND bash -c
        "rm *.txt ;
        ./inviscid_naca0012_reduced_order_consistency_snapshots.sh ${EXECUTABLE_OUTPUT_PATH} ${MPIMAX}
        return_val1=$? ;
        mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_reduced_order_consistency.prm
        return_val2=$? ;
        rm *.txt ;
        if [ $return_val1 -ne 0 ] || [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(INVISCID_NACA0012_REDUCED_ORDER_CONSISTENCY    REDUCED_ORDER
                                                                2D
                                                                PARALLEL
                                                                EULER
                                                                IMPLICIT
                                                                WEAK
                                                                UNCOLLOCATED
                                                                QUICK
                                                                INTEGRATION_TEST)
# =======================================
# Inviscid NACA0012 Adaptive Sampling
# =======================================
configure_file(inviscid_naca0012_adaptive_sampling.prm inviscid_naca0012_adaptive_sampling.prm COPYONLY)
add_test(
        NAME INVISCID_NACA0012_ADAPTIVE_SAMPLING
        COMMAND bash -c
        " 
        mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_adaptive_sampling.prm;
        mkdir -p NACA_rom_adapt_results;
        mv rom_table_iteration_final.txt ./NACA_rom_adapt_results ;
        mv snapshot_table_iteration_final.txt ./NACA_rom_adapt_results ;
        mv solution_snapshots_iteration_final.txt ./NACA_rom_adapt_results ; "
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(INVISCID_NACA0012_ADAPTIVE_SAMPLING    REDUCED_ORDER
                                                        2D
                                                        PARALLEL
                                                        EULER
                                                        IMPLICIT
                                                        WEAK
                                                        UNCOLLOCATED
                                                        EXTRA-LONG
                                                        INTEGRATION_TEST)

# =======================================
# Inviscid NACA0012 Adaptive Sampling Testing
# =======================================
#configure_file(inviscid_naca0012_adaptive_sampling_testing.prm inviscid_naca0012_adaptive_sampling_testing.prm COPYONLY)
#add_test(
#        NAME INVISCID_NACA0012_ADAPTIVE_SAMPLING_TESTING
#        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_adaptive_sampling_testing.prm
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

# =======================================
# Inviscid NACA0012 Snapshot Generator
# =======================================
#configure_file(inviscid_naca0012_snapshot_generator.sh inviscid_naca0012_snapshot_generator.sh COPYONLY)
#add_test(
#        NAME INVISCID_NACA0012_GENERATE_SNAPSHOTS
#        COMMAND bash -c
#        "./inviscid_naca0012_snapshot_generator.sh ${EXECUTABLE_OUTPUT_PATH}"
#        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

# =======================================
# Burgers Rewienski manufactured solution
# =======================================
configure_file(1d_burgers_rewienski_manufactured.prm 1d_burgers_rewienski_manufactured.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_REWIENSKI_MANUFACTURED_SOLUTION
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_manufactured.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_REWIENSKI_MANUFACTURED_SOLUTION REDUCED_ORDER
                                                            1D
                                                            SERIAL
                                                            BURGERS_REWIENSKI
                                                            IMPLICIT
                                                            WEAK
                                                            UNCOLLOCATED
                                                            CONVERGENCE
                                                            MODERATE
                                                            INTEGRATION_TEST)