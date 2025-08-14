set(TESTS_SOURCE
    rol_to_dealii_vector.cpp
    dealii_solver_rol_vector.cpp
    flow_constraints.cpp
    rol_objective.cpp
    full_space_step.cpp
    )

add_subdirectory(design_parameterization)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT OptimizationLib Optimization ${lib_spec})
    add_library(${OptimizationLib} STATIC ${TESTS_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${OptimizationLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT GridsLib Grids ${lib_spec})
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
    string(CONCAT ODESolverLib ODESolver ${lib_spec})
    string(CONCAT LinearSolverLib LinearSolver)
    string(CONCAT DesignParameterizationLib DesignParameterization ${lib_spec})
    target_link_libraries(${OptimizationLib} ${GridsLib})
    target_link_libraries(${OptimizationLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${OptimizationLib} ${ODESolverLib})
    target_link_libraries(${OptimizationLib} ${LinearSolverLib})
    target_link_libraries(${OptimizationLib} ${DesignParameterizationLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${OptimizationLib})
    endif()

    unset(OptimizationLib)
    unset(GridsLib)
    unset(DiscontinuousGalerkinLib)
    unset(ODESolverLib)
    unset(LinearSolverLib)
    unset(DesignParameterizationLib)
endforeach()
endforeach()

