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
target_link_libraries(Tests.exe ParametersLibrary)
target_link_libraries(Tests.exe ${LinearSolverLib})
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(Tests.exe)
endif()

ADD_TEST(NAME NNLS_known_tests
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> known)

ADD_TEST(NAME NNLS_matlab_test
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> matlab)

ADD_TEST(NAME NNLS_no_cols_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> noCols)

ADD_TEST(NAME NNLS_empty_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> empty)

ADD_TEST(NAME NNLS_random_problem
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> random)

ADD_TEST(NAME NNLS_zero_RHS
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> zeroRHS)

ADD_TEST(NAME NNLS_dependent_columns
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> depCols)

ADD_TEST(NAME NNLS_wide_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> wide)

ADD_TEST(NAME NNLS_zero_iter_w_sol
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> zeroIter)

ADD_TEST(NAME NNLS_n_iter
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> nIter)

ADD_TEST(NAME NNLS_max_iter
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> maxIter)


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
