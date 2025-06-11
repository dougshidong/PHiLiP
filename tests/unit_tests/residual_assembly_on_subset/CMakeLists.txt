set(TEST_SRC
    assemble_residual_on_subset.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_ASSEMBLE_RESIDUAL_ON_SUBSET)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    set(GridsLib Grids_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${GridsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else ()
        set(NMPI ${MPIMAX})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )
    if (dim EQUAL 1)
        set_tests_labels(${TEST_TARGET} RESIDUAL_ASSEMBLY_ON_SUBSET
                                        ${dim}D
                                        SERIAL
                                        QUICK
                                        UNIT_TEST)
    else ()
        set_tests_labels(${TEST_TARGET} RESIDUAL_ASSEMBLY_ON_SUBSET
                                        ${dim}D
                                        PARALLEL
                                        QUICK
                                        UNIT_TEST)
    endif()
    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(GridsLib)

endforeach()
