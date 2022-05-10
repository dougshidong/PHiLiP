
set(TESTS_SOURCE
    tests.cpp
    grid_study.cpp
    grid_refinement_study.cpp
    burgers_stability.cpp
    diffusion_exact_adjoint.cpp
    euler_gaussian_bump.cpp
    euler_gaussian_bump_enthalpy_check.cpp
    euler_naca0012.cpp
    euler_bump_optimization.cpp
    euler_naca0012_optimization.cpp
    shock_1d.cpp
#euler_gaussian_bump_adjoint.cpp
    euler_cylinder.cpp
    euler_cylinder_adjoint.cpp
    euler_vortex.cpp
    euler_entropy_waves.cpp
    advection_explicit_periodic.cpp
    convection_diffusion_explicit_periodic.cpp
    euler_split_inviscid_taylor_green_vortex.cpp
    optimization_inverse_manufactured/optimization_inverse_manufactured.cpp
    reduced_order.cpp
    reduced_order_pod_adaptation.cpp
    fd_state_sensitivity_wrt_parameter.cpp
    flow_solver_cases/flow_solver_case_base.cpp
    flow_solver_cases/periodic_cube_flow.cpp
    flow_solver_cases/periodic_turbulence.cpp
    flow_solver_cases/1D_burgers_rewienski_snapshot.cpp
    flow_solver_cases/1d_burgers_viscous_snapshot.cpp
    flow_solver.cpp
    dual_weighted_residual_mesh_adaptation.cpp
    taylor_green_vortex_energy_check.cpp
    taylor_green_vortex_restart_check.cpp
    flow_solver_cases/naca0012.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT TestsLib Tests_${dim}D)
    add_library(${TestsLib} STATIC ${TESTS_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TestsLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT GridsLib Grids_${dim}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT FunctionalLib Functional_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    string(CONCAT OptimizationLib Optimization_${dim}D)
    string(CONCAT LinearSolverLib LinearSolver)
    string(CONCAT GridRefinementLib GridRefinement_${dim}D)
    string(CONCAT PODLib POD_${dim}D)
    string(CONCAT InitialConditionsLib InitialConditions_${dim}D)
    target_link_libraries(${TestsLib} ${GridsLib})
    target_link_libraries(${TestsLib} ${NumericalFluxLib})
    target_link_libraries(${TestsLib} ${PhysicsLib})
    target_link_libraries(${TestsLib} ${FunctionalLib})
    target_link_libraries(${TestsLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TestsLib} ${ODESolverLib})
    target_link_libraries(${TestsLib} ${LinearSolverLib})
    target_link_libraries(${TestsLib} ${GridRefinementLib})
    target_link_libraries(${TestsLib} ${OptimizationLib})
    target_link_libraries(${TestsLib} ${PODLib})
    target_link_libraries(${TestsLib} ${InitialConditionsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TestsLib})
    endif()

    unset(GridsLib)
    unset(TestsLib)
    unset(DiscontinuousGalerkinLib)
    unset(FunctionalLib)
    unset(ODESolverLib)
    unset(NumericalFluxLib)
    unset(PhysicsLib)
    unset(LinearSolverLib)
    unset(GridRefinementLib)
    unset(OptimizationLib)
    unset(PODLib)
    unset(InitialConditionsLib)
endforeach()
