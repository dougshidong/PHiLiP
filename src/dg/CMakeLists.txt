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

set(POSSIBLE_N_SPECIES "0;11")

foreach(nspecies ${POSSIBLE_N_SPECIES})
foreach(dim RANGE 1 3)
    # Output library
    if(${nspecies} EQUAL "0")
        string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    else()
        string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D_${nspecies}-SPECIES)
    endif()
    add_library(${DiscontinuousGalerkinLib} STATIC ${DG_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${DiscontinuousGalerkinLib} PRIVATE PHILIP_DIM=${dim})
    target_compile_definitions(${DiscontinuousGalerkinLib} PRIVATE N_SPECIES=${nspecies})
    set_source_files_properties(${DG_SOURCE} LANGUAGE CXX)

    # Library dependency
    string(CONCAT SolutionLib Solution_${dim}D)
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    string(CONCAT PostprocessingLib Postprocessing_${dim}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT OperatorsLib Operator_Lib_${dim}D)
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
