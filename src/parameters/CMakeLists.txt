set(SOURCE
    parameters.cpp
    parameters_ode_solver.cpp
    parameters_linear_solver.cpp
    parameters_manufactured_convergence_study.cpp
    all_parameters.cpp
    )

set(ParameterLib ParametersLibrary)
add_library(${ParameterLib} STATIC ${SOURCE})

DEAL_II_SETUP_TARGET(${ParameterLib})

unset(ParameterLib)
