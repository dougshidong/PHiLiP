set(TEST_SRC
Auxiliary_variable_test.cpp
    )

foreach(dim RANGE 1 1)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_AUXILIARY_SOLUTION_TEST)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    target_link_libraries(${TEST_TARGET} ParametersLibrary)
   string(CONCAT OperatorsLib Operator_Lib_${dim}D)
   target_link_libraries(${TEST_TARGET} ${OperatorsLib})
    target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
#   # Setup target with deal.II
#   DEAL_II_SETUP_TARGET(${TEST_TARGET})
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
 #   unset(ParametersLib)
    unset(OperatorsLib)

endforeach()

set(TEST_SRC
Auxiliary_variable_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}AUXILIARY_SOLUTION_TEST)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    target_link_libraries(${TEST_TARGET} ParametersLibrary)
   string(CONCAT OperatorsLib Operator_Lib_${dim}D)
   target_link_libraries(${TEST_TARGET} ${OperatorsLib})
    target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
#   # Setup target with deal.II
#   DEAL_II_SETUP_TARGET(${TEST_TARGET})
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
 #   unset(ParametersLib)
    unset(OperatorsLib)

endforeach()
