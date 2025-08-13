set(ODE_SOURCE
    ode_solver_factory.cpp
    ode_solver_base.cpp
    runge_kutta_base.cpp
    runge_kutta_ode_solver.cpp
    low_storage_runge_kutta_ode_solver.cpp
    pod_galerkin_runge_kutta_ode_solver.cpp
    runge_kutta_methods/runge_kutta_methods.cpp
    runge_kutta_methods/rk_tableau_base.cpp
    runge_kutta_methods/low_storage_rk_tableau_base.cpp
    runge_kutta_methods/low_storage_runge_kutta_methods.cpp
    relaxation_runge_kutta/empty_RRK_base.cpp
    relaxation_runge_kutta/runge_kutta_store_entropy.cpp
    relaxation_runge_kutta/rrk_ode_solver_base.cpp
    relaxation_runge_kutta/algebraic_rrk_ode_solver.cpp
    relaxation_runge_kutta/root_finding_rrk_ode_solver.cpp
    implicit_ode_solver.cpp
    pod_galerkin_ode_solver.cpp
    pod_petrov_galerkin_ode_solver.cpp
    reduced_order_ode_solver.cpp
    JFNK_solver/jacobian_vector_product.cpp
    JFNK_solver/JFNK_solver.cpp
    hyper_reduced_petrov_galerkin_ode_solver.cpp)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT ODESolverLib ODESolver ${lib_spec})
    add_library(${ODESolverLib} STATIC ${ODE_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${ODESolverLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT HighOrderGridLib HighOrderGrid ${lib_spec})
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
    string(CONCAT LimiterLib Limiter ${lib_spec})
    string(CONCAT LinearSolverLib LinearSolver)
    string(CONCAT PODLib POD ${lib_spec})
    target_link_libraries(${ODESolverLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${ODESolverLib} ${LimiterLib})
    target_link_libraries(${ODESolverLib} ${HighOrderGridLib})
    target_link_libraries(${ODESolverLib} ${LinearSolverLib})
    target_link_libraries(${ODESolverLib} ${PODLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ODESolverLib})
    endif()


    unset(ODESolverLib)
    unset(DiscontinuousGalerkinLib)
    unset(LimiterLib)
    unset(HighOrderGridLib)

endforeach()
endforeach()
