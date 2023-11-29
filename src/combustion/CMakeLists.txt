set(SOURCE
    all_real_gas_constants.cpp
    allocatePointersLib.cpp
    NonEquilibriumVar.cpp
    ReactiveVar.cpp
    readspecies.cpp
    )

# Output library
set(ParameterLib ParametersLibrary)
add_library(${ParameterLib} STATIC ${SOURCE})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${ParameterLib})
endif()

unset(ParameterLib)
