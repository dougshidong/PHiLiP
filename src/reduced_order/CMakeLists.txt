set(SOURCE
    rbf_interpolation.cpp
    pod_basis_online.cpp
    reduced_order_solution.cpp
    rom_test_location.cpp
    pod_basis_offline.cpp
    halton.cpp
    halton_sampling.cpp
    nearest_neighbors.cpp
    min_max_scaler.cpp
    assemble_ECSW_residual.cpp
    assemble_ECSW_jacobian.cpp
    pod_adaptive_sampling.cpp
    hyper_reduced_adaptive_sampling.cpp
    adaptive_sampling_base.cpp
    assemble_ECSW_training_data_base.cpp
    hrom_test_location.cpp
    hyper_reduced_sampling_error_updated.cpp
    multi_core_helper_functions.cpp
    test_location_base.cpp)

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
