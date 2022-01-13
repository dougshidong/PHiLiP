set(SOURCE
        pod_basis.cpp
        pod_adaptation.cpp coarse_pod_basis.cpp coarse_pod_basis.h fine_pod_basis.cpp fine_pod_basis.h fine_not_in_coarse_basis.cpp fine_not_in_coarse_basis.h)

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