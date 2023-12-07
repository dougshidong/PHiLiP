set(SOURCE
    all_real_gas_constants.cpp
    allocatePointersLib.cpp
    NonEquilibriumVar.cpp
    ReactiveVar.cpp
    strManipLib.cpp
    readspecies.cpp
    )

# Output library
set(AllRealGasConstantsLib AllRealGasConstantsLibrary)
add_library(${AllRealGasConstantsLib} STATIC ${SOURCE})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${AllRealGasConstantsLib})
endif()

unset(AllRealGasConstantsLib)
