set(DG_SOURCE
    high_order_grid.cpp
    dg.cpp
    drdx.cpp
    weak_dg.cpp
    strong_dg.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    add_library(${DiscontinuousGalerkinLib} STATIC ${DG_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${DiscontinuousGalerkinLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT PostprocessingLib Postprocessing_${dim}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${DiscontinuousGalerkinLib} ${PostprocessingLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${NumericalFluxLib})
    target_link_libraries(${DiscontinuousGalerkinLib} ${PhysicsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${DiscontinuousGalerkinLib})
    endif()

    unset(DiscontinuousGalerkinLib)
    unset(NumericalFluxLib)
    unset(PhysicsLib)

endforeach()
