set(SOURCE
    parameters.cpp
    parameters_ode_solver.cpp
    parameters_linear_solver.cpp
    parameters_manufactured_convergence_study.cpp
    parameters_manufactured_solution.cpp
    parameters_euler.cpp
    parameters_navier_stokes.cpp
    parameters_grid_refinement_study.cpp
    parameters_functional.cpp
    parameters_grid_refinement.cpp
    all_parameters.cpp
        parameters_reduced_order.cpp)

# Output library
set(ParameterLib ParametersLibrary)
add_library(${ParameterLib} STATIC ${SOURCE})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${ParameterLib})
endif()

unset(ParameterLib)
