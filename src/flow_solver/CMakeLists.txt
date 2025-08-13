
set(FLOW_SOLVER_SOURCE
    flow_solver_cases/flow_solver_case_base.cpp
    flow_solver_cases/cube_flow_uniform_grid.cpp
    flow_solver_cases/periodic_cube_flow.cpp
    flow_solver_cases/non_periodic_cube_flow.cpp
    flow_solver_cases/periodic_turbulence.cpp
    flow_solver_cases/periodic_1D_unsteady.cpp
    flow_solver_cases/periodic_entropy_tests.cpp
    flow_solver_cases/1D_burgers_rewienski_snapshot.cpp
    flow_solver_cases/1d_burgers_viscous_snapshot.cpp
    flow_solver_cases/naca0012.cpp
    flow_solver_cases/gaussian_bump.cpp
    flow_solver_cases/limiter_convergence_tests.cpp
    flow_solver_cases/positivity_preserving_tests.cpp
    flow_solver.cpp
    flow_solver_factory.cpp)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT FlowSolverLib FlowSolver ${lib_spec})
    add_library(${FlowSolverLib} STATIC ${FLOW_SOLVER_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${FlowSolverLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1, 2, or 3 in the code
    target_compile_definitions(${FlowSolverLib} PRIVATE PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT GridsLib Grids ${lib_spec})
    string(CONCAT PhysicsLib Physics ${lib_spec})
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
    string(CONCAT FunctionalLib Functional ${lib_spec})
    string(CONCAT ODESolverLib ODESolver ${lib_spec})
    string(CONCAT InitialConditionsLib InitialConditions ${lib_spec})
    string(CONCAT PODLib POD ${lib_spec})
    string(CONCAT MeshAdaptationLib MeshAdaptation ${lib_spec})
    target_link_libraries(${FlowSolverLib} ${GridsLib})
    target_link_libraries(${FlowSolverLib} ${PhysicsLib})
    target_link_libraries(${FlowSolverLib} ${FunctionalLib})
    target_link_libraries(${FlowSolverLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${FlowSolverLib} ${ODESolverLib})
    target_link_libraries(${FlowSolverLib} ${InitialConditionsLib})
    target_link_libraries(${FlowSolverLib} ${PODLib})
    target_link_libraries(${FlowSolverLib} ${MeshAdaptationLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${FlowSolverLib})
    endif()

    unset(GridsLib)
    unset(FlowSolverLib)
    unset(DiscontinuousGalerkinLib)
    unset(FunctionalLib)
    unset(ODESolverLib)
    unset(PhysicsLib)
    unset(InitialConditionsLib)
    unset(PODLib)
    unset(MeshAdaptationLib)
endforeach()
endforeach()