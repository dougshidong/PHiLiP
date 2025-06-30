set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB MY_FILES "*.csv")
file(COPY ${MY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_burgers_rewienski_build_NNLS_problem.prm 1d_burgers_rewienski_build_NNLS_problem.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_NNLS_PROBLEM_ASSEMBLY
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_build_NNLS_problem.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_NNLS_PROBLEM_ASSEMBLY   HYPER_REDUCTION
                                                    1D
                                                    SERIAL
                                                    BURGERS_REWIENSKI
                                                    IMPLICIT
                                                    WEAK
                                                    UNCOLLOCATED
                                                    MODERATE
                                                    INTEGRATION_TEST)

configure_file(1d_burgers_rewienski_hyper_post_sampling.prm 1d_burgers_rewienski_hyper_post_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_POST_SAMPLING
        COMMAND bash -c
        "
        if [ -d  ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_one_param ]; then
                mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_post_sampling.prm;
                cp weights_table_iteration_HROM_post_sampling.txt ./1d_burgers_one_param ;
        elif [ -d  ${CMAKE_CURRENT_BINARY_DIR}/../reduced_order/1d_burgers_one_param ]; then
                cp -r ${CMAKE_CURRENT_BINARY_DIR}/../reduced_order/1d_burgers_one_param ${CMAKE_CURRENT_BINARY_DIR}/;
                mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_post_sampling.prm;
                cp weights_table_iteration_HROM_post_sampling.txt ./1d_burgers_one_param ;
        else
                echo 'Run 1D_BURGERS_REWIENSKI_ONE_PARAM_ADAPTIVE_SAMPLING test first; ensure folder 1d_burgers_one_param exists in the reduced_order folder';
                exit 1;
        fi"
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_NNLS_PROBLEM_ASSEMBLY   HYPER_REDUCTION
                                                    1D
                                                    SERIAL
                                                    BURGERS_REWIENSKI
                                                    IMPLICIT
                                                    WEAK
                                                    UNCOLLOCATED
                                                    MODERATE
                                                    INTEGRATION_TEST)

configure_file(1d_burgers_rewienski_hyper_reduction_test.prm 1d_burgers_rewienski_hyper_reduction_test.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_REDUCTION_TEST
        COMMAND bash -c
        " 
        if [ -f  ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_one_param/weights_table_iteration_HROM_post_sampling.txt ]; then
                mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_reduction_test.prm     
        else
                echo 'Run 1D_BURGERS_REWIENSKI_ONE_PARAM_ADAPTIVE_SAMPLING and 1D_BURGERS_HYPER_POST_SAMPLING tests first; ensure folder 1d_burgers_one_param exists in the reduced_order folder';
                exit 1;
        fi"  
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_HYPER_REDUCTION_TEST    HYPER_REDUCTION
                                                    1D
                                                    SERIAL
                                                    BURGERS_REWIENSKI
                                                    IMPLICIT
                                                    WEAK
                                                    UNCOLLOCATED
                                                    MODERATE
                                                    INTEGRATION_TEST)

configure_file(1d_burgers_rewienski_hyper_adaptive_sampling.prm 1d_burgers_rewienski_hyper_adaptive_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_ADAPTIVE_SAMPLING_TEST
        COMMAND bash -c
        "mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_adaptive_sampling.prm
        mkdir -p 1d_burgers_hyper_adapt_results;
        mv rom_table_iteration_final.txt ./1d_burgers_hyper_adapt_results ;
        mv snapshot_table_iteration_final.txt ./1d_burgers_hyper_adapt_results ;
        mv solution_snapshots_iteration_final.txt ./1d_burgers_hyper_adapt_results ;
        mv weights_table_iteration_final.txt ./1d_burgers_hyper_adapt_results ; "
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_HYPER_ADAPTIVE_SAMPLING_TEST    HYPER_REDUCTION
                                                            1D
                                                            SERIAL
                                                            BURGERS_REWIENSKI
                                                            IMPLICIT
                                                            WEAK
                                                            UNCOLLOCATED
                                                            MODERATE
                                                            INTEGRATION_TEST)

configure_file(1d_burgers_rewienski_hyper_sampling_new_error.prm 1d_burgers_rewienski_hyper_sampling_new_error.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_SAMPLING_NEW_ERROR
        COMMAND bash -c
        "mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_sampling_new_error.prm
        mkdir -p 1d_burgers_hyper_new_error_results;
        mv rom_table_iteration_final.txt ./1d_burgers_hyper_new_error_results ;
        mv snapshot_table_iteration_final.txt ./1d_burgers_new_error_results ;
        mv solution_snapshots_iteration_final.txt ./1d_burgers_hyper_new_error_results ;
        mv weights_table_iteration_final.txt ./1d_burgers_hyper_new_error_results ; "
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_BURGERS_HYPER_POST_SAMPLING HYPER_REDUCTION
                                                1D
                                                SERIAL
                                                BURGERS_REWIENSKI
                                                IMPLICIT
                                                WEAK
                                                UNCOLLOCATED
                                                MODERATE
                                                INTEGRATION_TEST)

# =======================================
# Inviscid NACA0012 Adaptive Sampling
# =======================================

# Tests the same code as the Burgers' post sampling test, but takes much longer as it samples at 400 points
# configure_file(inviscid_naca0012_hyper_post_sampling.prm inviscid_naca0012_hyper_post_sampling.prm COPYONLY)
# add_test(
#         NAME INVISCID_NACA0012_HYPER_POST_SAMPLING
#         COMMAND bash -c 
#         "if [ -d  ${CMAKE_CURRENT_BINARY_DIR}/NACA_rom_adapt_results ]; then
#                 mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_hyper_post_sampling.prm;
#         elif [ -d  ${CMAKE_CURRENT_BINARY_DIR}/../reduced_order/NACA_rom_adapt_results ]; then
#                 cp -r ${CMAKE_CURRENT_BINARY_DIR}/../reduced_order/NACA_rom_adapt_results ${CMAKE_CURRENT_BINARY_DIR}/;
#                 mpirun -np  1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_hyper_post_sampling.prm; 
#         else
#                 echo 'Run INVISCID_NACA0012_ADAPTIVE_SAMPLING test first; ensure folder NACA_rom_adapt_results exists in the reduced_order folder';
#                 exit 1;
#         fi"
#         WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )
# set_tests_labels(INVISCID_NACA0012_HYPER_POST_SAMPLING HYPER_REDUCTION
#                                                 2D
#                                                 SERIAL
#                                                 EULER
#                                                 IMPLICIT
#                                                 WEAK
#                                                 UNCOLLOCATED
#                                                 EXTRA-LONG
#                                                 INTEGRATION_TEST)

# The following test can only be run on Narval currently with a 2009G node as they require a large amount of memory
# configure_file(inviscid_naca0012_hyper_adaptive_sampling.prm inviscid_naca0012_hyper_adaptive_sampling.prm COPYONLY)
# add_test(
#         NAME INVISCID_NACA0012_HYPER_ADAPTIVE_SAMPLING_RUN
#         COMMAND bash -c
#         "
#         mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_hyper_adaptive_sampling.prm
#         mkdir -p NACA_hyper_adapt_results;
#         mv rom_table_iteration_final.txt ./NACA_hyper_adapt_results ;
#         mv snapshot_table_iteration_final.txt ./NACA_hyper_adapt_results ;
#         mv solution_snapshots_iteration_final.txt ./NACA_hyper_adapt_results ;
#         mv weights_table_iteration_final.txt ./NACA_hyper_adapt_results ; "
#         WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )

# Uncomment the following two tests if you would like to find the "true" error between the models built adaptively with and without hyperreduction
# Running them are expensive as they calculate the error at 400 parameter locations, and do not test any additional areas of the reduced order code
# configure_file(ROM_error_NACA.prm ROM_error_NACA.prm  COPYONLY)
# add_test(
#         NAME ROM_ERROR_NACA
#         COMMAND bash -c
#         "if [ -d  ${CMAKE_CURRENT_BINARY_DIR}/NACA_rom_adapt_results ]; then
#                 mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/ROM_error_NACA.prm;
#         elif [ -d  ${CMAKE_CURRENT_BINARY_DIR}/../reduced_order/NACA_rom_adapt_results ]; then
#                 cp -r ${CMAKE_CURRENT_BINARY_DIR}/../reduced_order/NACA_rom_adapt_results ${CMAKE_CURRENT_BINARY_DIR}/;
#                 mpirun -np  1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/ROM_error_NACA.prm;
#         else
#                 echo 'Run INVISCID_NACA0012_ADAPTIVE_SAMPLING test first; ensure folder NACA_rom_adapt_results exists in the reduced_order folder';
#                 exit 1;
#         fi"
#         WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )

# configure_file(HROM_error_NACA.prm HROM_error_NACA.prm  COPYONLY)
# add_test(
#         NAME HROM_ERROR_NACA
#         COMMAND bash -c
#         "if [ -d  ${CMAKE_CURRENT_BINARY_DIR}/NACA_hyper_adapt_results ]; then
#                 mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/HROM_error_NACA.prm;
#         else
#                 echo 'Run INVISCID_NACA0012_HYPER_ADAPTIVE_SAMPLING_RUN test first; ensure folder NACA_hyper_adapt_results exists';
#                 exit 1;
#         fi"
#         WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )

configure_file(inviscid_naca0012_one_param_multi.prm inviscid_naca0012_one_param_multi.prm COPYONLY)
add_test(
        NAME INVISCID_NACA0012_ONE_MULTI
        COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_one_param_multi.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(INVISCID_NACA0012_ONE_MULTI            HYPER_REDUCTION
                                                        2D
                                                        PARALLEL
                                                        EULER
                                                        IMPLICIT
                                                        WEAK
                                                        UNCOLLOCATED
                                                        EXTRA-LONG
                                                        INTEGRATION_TEST)