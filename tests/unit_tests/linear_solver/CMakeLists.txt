# Run tests in parallel iff MPI enabled 
IF (MYAPP_MPI)
  set(MPIGO /usr/bin/mpiexec -np 1)
ELSE()
  set(MPIGO  )
ENDIF()

INCLUDE_DIRECTORIES ( ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
MESSAGE("   Eigen_Dir = ${EIGEN3_INCLUDE_DIR}")
LINK_DIRECTORIES ( ${EIGEN3_INCLUDE_DIR} )
LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

file(GLOB MY_FILES "*.csv")
file(COPY ${MY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE(Tests.exe NNLS_tests.cpp)
string(CONCAT LinearSolverLib LinearSolver)
string(CONCAT PODLib POD_1D)
target_link_libraries(Tests.exe ParametersLibrary)
target_link_libraries(Tests.exe ${LinearSolverLib})
target_link_libraries(Tests.exe ${PODLib})
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(Tests.exe)
endif()

ADD_TEST(NAME NNLS_known_tests
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> known)
set_tests_labels(NNLS_known_tests   LINEAR_SOLVER
                                    SERIAL
                                    QUICK
                                    UNIT_TEST)

ADD_TEST(NAME NNLS_matlab_test
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> matlab)
set_tests_labels(NNLS_matlab_test   LINEAR_SOLVER
                                    SERIAL
                                    QUICK
                                    UNIT_TEST)

ADD_TEST(NAME NNLS_no_cols_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> noCols)
set_tests_labels(NNLS_no_cols_matrix    LINEAR_SOLVER
                                        SERIAL
                                        QUICK
                                        UNIT_TEST)

ADD_TEST(NAME NNLS_empty_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> empty)
set_tests_labels(NNLS_empty_matrix  LINEAR_SOLVER
                                    SERIAL
                                    QUICK
                                    UNIT_TEST)

ADD_TEST(NAME NNLS_random_problem
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> random)
set_tests_labels(NNLS_random_problem    LINEAR_SOLVER
                                        SERIAL
                                        QUICK
                                        UNIT_TEST)

ADD_TEST(NAME NNLS_zero_RHS
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> zeroRHS)
set_tests_labels(NNLS_zero_RHS  LINEAR_SOLVER
                                SERIAL
                                QUICK
                                UNIT_TEST)

ADD_TEST(NAME NNLS_dependent_columns
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> depCols)
set_tests_labels(NNLS_dependent_columns LINEAR_SOLVER
                                        SERIAL
                                        QUICK
                                        UNIT_TEST)

ADD_TEST(NAME NNLS_wide_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> wide)
set_tests_labels(NNLS_wide_matrix   LINEAR_SOLVER
                                    SERIAL
                                    QUICK
                                    UNIT_TEST)

ADD_TEST(NAME NNLS_zero_iter_w_sol
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> zeroIter)
set_tests_labels(NNLS_zero_iter_w_sol   LINEAR_SOLVER
                                        SERIAL
                                        QUICK
                                        UNIT_TEST)

ADD_TEST(NAME NNLS_n_iter
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> nIter)
set_tests_labels(NNLS_n_iter    LINEAR_SOLVER
                                SERIAL
                                QUICK
                                UNIT_TEST)

ADD_TEST(NAME NNLS_max_iter
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> maxIter)
set_tests_labels(NNLS_max_iter  LINEAR_SOLVER
                                SERIAL
                                QUICK
                                UNIT_TEST)

ADD_TEST(NAME NNLS_multi_core
COMMAND mpirun -n ${MPIMAX} $<TARGET_FILE:Tests.exe> multiCore)
set_tests_labels(NNLS_multi_core    LINEAR_SOLVER
                                    PARALLEL
                                    QUICK
                                    UNIT_TEST)

#set(TEST_SRC
#	NNLS_tests.cpp)
#
#    # Output executable
#    string(CONCAT TEST_TARGET NNLS_known_tests)
#    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
#    add_executable(${TEST_TARGET} ${TEST_SRC})
#    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
#    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=1)
#
#    # Compile this executable when 'make unit_tests'
#    add_dependencies(unit_tests ${TEST_TARGET})
# #   add_dependencies(${dim}D ${TEST_TARGET})
#
#    # Library dependency
#    target_link_libraries(${TEST_TARGET} LinearSolverLib)
#    
#    # Setup target with deal.II
#    if (NOT DOC_ONLY)
#        DEAL_II_SETUP_TARGET(${TEST_TARGET})
#    endif()
#
#    add_test(
#      NAME ${TEST_TARGET}
#      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
#      WORKING_DIRECTORY ${TEST_OUTPUT_DIR})
#
#    unset(TEST_TARGET)
#	unset(LinearSolverLib)
