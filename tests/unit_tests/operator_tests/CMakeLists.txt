set(TEST_SRC
    Volume_operators_test.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_VOLUME_OPERATORS)
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

set(TEST_SRC
    GCL_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_GCL)
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

set(TEST_SRC
    Surface_oper_test.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_SURFACE_OPERATORS)
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

set(TEST_SRC
Surface_GCL_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_SURFACE_GCL)
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

set(TEST_SRC
Surface_GCL_Superparametric_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_SURFACE_GCL_SUPERPARAMETRIC)
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

set(TEST_SRC
GCL_Collocated_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_GCL_COLLOCATED)
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

set(TEST_SRC
Flux_oper_test.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_FLUX_OPERATORS)
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

set(TEST_SRC
    chain_rule_test.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}CHAIN_RULE_TEST)
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

set(TEST_SRC
weighted_int_by_parts.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}WEIGHT_INT_BY_PARTS_TEST)
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

set(TEST_SRC
Metric_split_form_gradient.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}METRIC_GRADIENT_TEST)
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

set(TEST_SRC
Surface_conforming_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}SURFACE_CONFORMING_TEST)
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

set(TEST_SRC
Weight_adjusted_mass_inverse_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}WEIGHT_ADJUSTED_MASS_INV_TEST)
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
