set(ODE_SOURCE
    ode_solver_factory.cpp
    ode_solver_base.cpp
    explicit_ode_solver.cpp
    implicit_ode_solver.cpp
    pod_galerkin_ode_solver.cpp
    pod_petrov_galerkin_ode_solver.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    add_library(${ODESolverLib} STATIC ${ODE_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${ODESolverLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT LinearSolverLib LinearSolver)
    target_link_libraries(${ODESolverLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${ODESolverLib} ${HighOrderGridLib})
    target_link_libraries(${ODESolverLib} ${LinearSolverLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ODESolverLib})
    endif()


    unset(ODESolverLib)
    unset(DiscontinuousGalerkinLib)
    unset(HighOrderGridLib)

endforeach()
