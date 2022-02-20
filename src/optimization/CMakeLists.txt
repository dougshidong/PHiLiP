set(TESTS_SOURCE
    rol_to_dealii_vector.cpp
    dealii_solver_rol_vector.cpp
    flow_constraints.cpp
    rol_objective.cpp
    full_space_step.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT OptimizationLib Optimization_${dim}D)
    add_library(${OptimizationLib} STATIC ${TESTS_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${OptimizationLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT GridsLib Grids_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    string(CONCAT LinearSolverLib LinearSolver)
    target_link_libraries(${OptimizationLib} ${GridsLib})
    target_link_libraries(${OptimizationLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${OptimizationLib} ${ODESolverLib})
    target_link_libraries(${OptimizationLib} ${LinearSolverLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${OptimizationLib})
    endif()

    unset(OptimizationLib)
    unset(GridsLib)
    unset(DiscontinuousGalerkinLib)
    unset(ODESolverLib)
    unset(LinearSolverLib)

endforeach()

