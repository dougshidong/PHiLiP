set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB MY_FILES "*.csv")
file(COPY ${MY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Test Burgers Rewienski Galerkin and Petrov-Galerkin reduced-order solvers
# =======================================
#Be careful when using bash in cmake, the test will pass if the only last command executed successfully (i.e. if the last command is rm *.txt, the test will always pass)
# configure_file(1d_burgers_rewienski_reduced_order_consistency_snapshots.sh 1d_burgers_rewienski_reduced_order_consistency_snapshots.sh COPYONLY)
# configure_file(1d_burgers_rewienski_reduced_order_consistency.prm 1d_burgers_rewienski_reduced_order_consistency.prm COPYONLY)
# add_test(
#         NAME 1D_BURGERS_NNLS_PROBLEM_ASSEMBLY
#         COMMAND bash -c
#         "rm *.txt ;
#         ./1d_burgers_rewienski_reduced_order_consistency_snapshots.sh ${EXECUTABLE_OUTPUT_PATH}
#         return_val1=$? ;
#         mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_reduced_order_consistency.prm ;
#         return_val2=$? ;
#         rm *.txt ;
#         if [ $return_val1 -ne 0 ] || [ $return_val2 -ne 0 ]; then exit 1; else exit 0; fi"
#         WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )

configure_file(1d_burgers_rewienski_reduced_order_consistency_snapshots.sh 1d_burgers_rewienski_reduced_order_consistency_snapshots.sh COPYONLY)
configure_file(1d_burgers_rewienski_build_NNLS_problem.prm 1d_burgers_rewienski_build_NNLS_problem.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_NNLS_PROBLEM_ASSEMBLY
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_build_NNLS_problem.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_burgers_rewienski_hyper_reduction_test.prm 1d_burgers_rewienski_hyper_reduction_test.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_REDUCTION_TEST
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_reduction_test.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_burgers_rewienski_hyper_adaptive_sampling.prm 1d_burgers_rewienski_hyper_adaptive_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_ADAPTIVE_SAMPLING_TEST
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_adaptive_sampling.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_burgers_rewienski_hyper_post_sampling.prm 1d_burgers_rewienski_hyper_post_sampling.prm COPYONLY)
add_test(
        NAME 1D_BURGERS_HYPER_POST_SAMPLING
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_burgers_rewienski_hyper_post_sampling.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(inviscid_naca0012_hyper_post_sampling.prm inviscid_naca0012_hyper_post_sampling.prm COPYONLY)
add_test(
        NAME INVISCID_NACA0012_HYPER_POST_SAMPLING
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_hyper_post_sampling.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(inviscid_naca0012_hyper_adaptive_sampling.prm inviscid_naca0012_hyper_adaptive_sampling.prm COPYONLY)
add_test(
        NAME INVISCID_NACA0012_HYPER_ADAPTIVE_SAMPLING_TEST
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/inviscid_naca0012_hyper_adaptive_sampling.prm
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

