set(DG_SOURCE
    dg_factory.cpp
    dg_base.cpp
    dg_base_state.cpp
    residual_sparsity_patterns.cpp
    weak_dg.cpp
    strong_dg.cpp
    artificial_dissipation.cpp
    artificial_dissipation_factory.cpp
    )

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
    add_library(${DiscontinuousGalerkinLib} STATIC ${DG_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${DiscontinuousGalerkinLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1, 2, or 3 in the code
    target_compile_definitions(${DiscontinuousGalerkinLib} PRIVATE PHILIP_SPECIES=${species})
    set_source_files_properties(${DG_SOURCE} LANGUAGE CXX)

    # Library dependency
    string(CONCAT SolutionLib Solution ${lib_spec})
    string(CONCAT HighOrderGridLib HighOrderGrid ${lib_spec})
    string(CONCAT PostprocessingLib Postprocessing ${lib_spec})
    string(CONCAT NumericalFluxLib NumericalFlux ${lib_spec})
    string(CONCAT PhysicsLib Physics ${lib_spec})
    string(CONCAT OperatorsLib Operator_Lib ${lib_spec})
    target_link_libraries(${DiscontinuousGalerkinLib} ${SolutionLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${HighOrderGridLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${PostprocessingLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${NumericalFluxLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${PhysicsLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${OperatorsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${DiscontinuousGalerkinLib})
    endif()

    unset(SolutionLib)
    unset(HighOrderGridLib)
    unset(PostprocessingLib)
    unset(DiscontinuousGalerkinLib)
    unset(NumericalFluxLib)
    unset(PhysicsLib)
    unset(OperatorsLib)
endforeach()
endforeach()
