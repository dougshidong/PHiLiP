
set(FLOW_SOLVER_SOURCE
    flow_solver_cases/flow_solver_case_base.cpp
    flow_solver_cases/periodic_cube_flow.cpp
    flow_solver_cases/non_periodic_cube_flow.cpp
    flow_solver_cases/periodic_turbulence.cpp
    flow_solver_cases/periodic_1D_unsteady.cpp
    flow_solver_cases/periodic_entropy_tests.cpp
    flow_solver_cases/1D_burgers_rewienski_snapshot.cpp
    flow_solver_cases/1d_burgers_viscous_snapshot.cpp
    flow_solver_cases/naca0012.cpp
    flow_solver_cases/gaussian_bump.cpp
    flow_solver.cpp
    flow_solver_factory.cpp)

set(POSSIBLE_N_SPECIES "0;2;5;9;11")

foreach(nspecies ${POSSIBLE_N_SPECIES})
foreach(dim RANGE 1 3)
    # Output library
    if(${nspecies} EQUAL "0")
        string(CONCAT FlowSolverLib FlowSolver_${dim}D)
    else()
        string(CONCAT FlowSolverLib FlowSolver_${dim}D_${nspecies}-SPECIES)
    endif()
    add_library(${FlowSolverLib} STATIC ${FLOW_SOLVER_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${FlowSolverLib} PRIVATE PHILIP_DIM=${dim})
    target_compile_definitions(${FlowSolverLib} PRIVATE N_SPECIES=${nspecies})

    # Library dependency
    string(CONCAT GridsLib Grids_${dim}D)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT FunctionalLib Functional_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    string(CONCAT InitialConditionsLib InitialConditions_${dim}D)
    string(CONCAT PODLib POD_${dim}D)
    string(CONCAT MeshAdaptationLib MeshAdaptation_${dim}D)
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