set(SOURCE
    pod_basis.cpp
    pod_basis_types.cpp
    pod_adaptation.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT PODLib POD_${dim}D)
    add_library(${PODLib} STATIC ${SOURCE})

    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${PODLib} ${ParameterLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${PODLib})
    endif()

    target_compile_definitions(${PODLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    unset(PODLib)

endforeach()