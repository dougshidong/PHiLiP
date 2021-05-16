set(TEST_SRC
    assemble_d2R_timing.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_assemble_d2R_timing)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        # Only 4 cells, so more than that and we start having
        # trouble with parallelism
        if (${MPIMAX} GREATER 4)
            set(NMPI 4)
        else()
            set(NMPI ${MPIMAX})
        endif()
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(DiscontinuousGalerkinLib)
    unset(ODESolverLib)

endforeach()
set(TEST_SRC
    dRdX_fd_vs_ad.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_dRdX_fd_vs_ad)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        set(NMPI ${MPIMAX})
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    d2RdWdX_fd_vs_ad.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_d2RdWdX_fd_vs_ad)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        # Only 4 cells, so more than that and we start having
        # trouble with parallelism
        if (${MPIMAX} GREATER 4)
            set(NMPI 4)
        else()
            set(NMPI ${MPIMAX})
        endif()
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(DiscontinuousGalerkinLib)
    unset(ODESolverLib)

endforeach()

set(TEST_SRC
    d2RdWdW_fd_vs_ad.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_d2RdWdW_fd_vs_ad)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        # Only 4 cells, so more than that and we start having
        # trouble with parallelism
        if (${MPIMAX} GREATER 4)
            set(NMPI 4)
        else()
            set(NMPI ${MPIMAX})
        endif()
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(DiscontinuousGalerkinLib)
    unset(ODESolverLib)

endforeach()

set(TEST_SRC
    d2RdXdX_fd_vs_ad.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_d2RdXdX_fd_vs_ad)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        # Only 4 cells, so more than that and we start having
        # trouble with parallelism
        if (${MPIMAX} GREATER 4)
            set(NMPI 4)
        else()
            set(NMPI ${MPIMAX})
        endif()
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    dRdW_fd_vs_ad.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_dRdW_fd_vs_ad)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        set(NMPI ${MPIMAX})
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    compare_rhs.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_compare_rhs)
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

    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(GridsLib)

endforeach()

set(TEST_SRC
    check_symmetric_hessian.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_check_symmetric_hessian)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
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

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    symmetric_functional_hessian.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_symmetric_functional_hessian)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT FunctionalLib Functional_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${FunctionalLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
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

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    symmetric_KKT_hessian.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_symmetric_KKT_hessian)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT FunctionalLib Functional_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${FunctionalLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
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

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    dXvdXs.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_dXvdXs)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        set(NMPI ${MPIMAX})
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(HighOrderGridLib)

endforeach()

# This test currently fails.
# For some reason the constrained_linear_operator or application of constraints only
# works correctly in parallel as mentionned in commit 5410626f740a52be96b3df05b40da0d6efe3e391
foreach(dim RANGE 2 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_dXvdXs)

    set(NMPI 1)
    add_test(
      NAME ${TEST_TARGET}_serial
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)

endforeach()

set(TEST_SRC
    solve_KKT.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_solve_KKT)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} Functional_${dim}D)
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
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

    unset(TEST_TARGET)
    unset(ParametersLib)

endforeach()
