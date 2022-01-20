set(FLOW_SOLVER_CASES_SOURCE
        periodic_cube_flow.cpp
        )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT FlowSolverCasesLib FlowSolverCases_${dim}D)
    add_library(${FlowSolverCasesLib} STATIC ${FLOW_SOLVER_CASES_SOURCE})
    # Link with TestsLib
    string(CONCAT TestsLib Tests_${dim}D)
    target_link_libraries(${FlowSolverCasesLib} ${TestsLib})

    target_compile_definitions(${FlowSolverCasesLib} PRIVATE PHILIP_DIM=${dim})
    unset(TestsLib)

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${FlowSolverCasesLib})
    endif()

    unset(FlowSolverCases)
endforeach()