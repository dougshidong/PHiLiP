set(TEST_SRC
    msh_out_test.cpp
    )

foreach(dim RANGE 1 3)
    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_msh_out_test)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT GridRefinementLib GridRefinement_${dim}D)
    target_link_libraries(${TEST_TARGET} ${GridRefinementLib})
    # Setup target with deal.II
    DEAL_II_SETUP_TARGET(${TEST_TARGET})

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(GridRefinementLib)
endforeach()
