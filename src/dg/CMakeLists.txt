set(DG_SOURCE
    dg_factory.cpp
    dg.cpp
    residual_sparsity_patterns.cpp
    weak_dg.cpp
    strong_dg.cpp
    artificial_dissipation.cpp
    artificial_dissipation_factory.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    add_library(${DiscontinuousGalerkinLib} STATIC ${DG_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${DiscontinuousGalerkinLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    string(CONCAT PostprocessingLib Postprocessing_${dim}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT OperatorsLib Operator_Lib_${dim}D)
    target_link_libraries(${DiscontinuousGalerkinLib} ${HighOrderGridLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${PostprocessingLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${NumericalFluxLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${PhysicsLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${OperatorsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${DiscontinuousGalerkinLib})
    endif()

    unset(HighOrderGridLib)
    unset(PostprocessingLib)
    unset(DiscontinuousGalerkinLib)
    unset(NumericalFluxLib)
    unset(PhysicsLib)
    unset(OperatorsLib)

endforeach()
