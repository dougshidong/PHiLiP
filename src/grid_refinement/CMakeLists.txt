set(GRID_REFINEMENT_SOURCE
    gmsh_out.cpp
    )

# Output library
foreach(dim RANGE 1 3)
    string(CONCAT GridRefinementLib GridRefinement_${dim}D)
    add_library(${GridRefinementLib} STATIC ${GRID_REFINEMENT_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${GridRefinementLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${GridRefinementLib})
    endif()

    unset(GridRefinementLib)

endforeach()