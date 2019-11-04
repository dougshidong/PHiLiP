set(SOURCE
    parameters.cpp
    parameters_ode_solver.cpp
    parameters_linear_solver.cpp
    parameters_manufactured_convergence_study.cpp
    parameters_euler.cpp
    all_parameters.cpp
    )

# Output library
set(ParameterLib ParametersLibrary)
add_library(${ParameterLib} STATIC ${SOURCE})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${ParameterLib})
endif()

unset(ParameterLib)
