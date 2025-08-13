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
    TGV_scaling.cpp
    optimization_inverse_manufactured/optimization_inverse_manufactured.cpp
    dual_weighted_residual_mesh_adaptation.cpp
    anisotropic_mesh_adaptation_cases.cpp
    taylor_green_vortex_energy_check.cpp
    taylor_green_vortex_restart_check.cpp
    pod_adaptive_sampling_run.cpp
    pod_adaptive_sampling_testing.cpp
    reduced_order.cpp
    unsteady_reduced_order.cpp
    rom_import_helper_functions.cpp
    time_refinement_study.cpp
    time_refinement_study_reference.cpp
    h_refinement_study_isentropic_vortex.cpp
    rrk_numerical_entropy_conservation_check.cpp
    euler_entropy_conserving_split_forms_check.cpp
    homogeneous_isotropic_turbulence_initialization_check.cpp
    khi_robustness.cpp
    build_NNLS_problem.cpp
    hyper_reduction_comparison.cpp
    hyper_adaptive_sampling_run.cpp
    hyper_reduction_post_sampling.cpp
    ROM_error_post_sampling.cpp
    HROM_error_post_sampling.cpp
    hyper_adaptive_sampling_new_error.cpp
    halton_sampling_run.cpp
    bound_preserving_limiter_tests.cpp
    naca0012_unsteady_check_quick.cpp
    real_gas_vs_euler_primitive_to_conservative_check.cpp
    euler_vortex_advection_error_study.cpp
    )

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT TestsLib Tests ${lib_spec})

    add_library(${TestsLib} STATIC ${TESTS_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TestsLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1, 2, or 3 in the code
    target_compile_definitions(${TestsLib} PRIVATE PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT GridsLib Grids ${lib_spec})
    string(CONCAT NumericalFluxLib NumericalFlux ${lib_spec})
    string(CONCAT PhysicsLib Physics ${lib_spec})
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
    string(CONCAT FunctionalLib Functional ${lib_spec})
    string(CONCAT ODESolverLib ODESolver ${lib_spec})
    string(CONCAT OptimizationLib Optimization ${lib_spec})
    string(CONCAT LinearSolverLib LinearSolver)
    string(CONCAT GridRefinementLib GridRefinement ${lib_spec})
    string(CONCAT PODLib POD ${lib_spec})
    string(CONCAT InitialConditionsLib InitialConditions ${lib_spec})
    string(CONCAT FlowSolverLib FlowSolver ${lib_spec})
    string(CONCAT ExactSolutionsLib ExactSolutions ${lib_spec})
    string(CONCAT MeshAdaptationLib MeshAdaptation ${lib_spec})
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
    target_link_libraries(${TestsLib} ${FlowSolverLib})
    target_link_libraries(${TestsLib} ${ExactSolutionsLib})
    target_link_libraries(${TestsLib} ${MeshAdaptationLib})
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
    unset(FlowSolverLib)
    unset(ExactSolutionsLib)
    unset(MeshAdaptationLib)
endforeach()
endforeach()