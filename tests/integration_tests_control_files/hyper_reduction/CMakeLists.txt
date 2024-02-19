set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB MY_FILES "*.csv")
file(COPY ${MY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

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

configure_file(ROM_error_NACA.prm ROM_error_NACA.prm  COPYONLY)
add_test(
        NAME ROM_ERROR_NACA
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/ROM_error_NACA.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(HROM_error_NACA.prm HROM_error_NACA.prm  COPYONLY)
add_test(
        NAME HROM_ERROR_NACA
        COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/HROM_error_NACA.prm 
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
