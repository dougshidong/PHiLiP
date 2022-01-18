add_subdirectory(grids)

set(GRID_SOURCE
    high_order_grid.cpp
    gmsh_reader.cpp
    meshmover_linear_elasticity.cpp
    free_form_deformation.cpp
    mesh_adaptation.cpp
    mesh_error_estimate.cpp
    dual_weighted_residual.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    add_library(${HighOrderGridLib} STATIC ${GRID_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${HighOrderGridLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${HighOrderGridLib})
    endif()

    unset(HighOrderGridLib)

endforeach()
