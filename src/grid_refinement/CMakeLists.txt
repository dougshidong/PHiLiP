set(GRID_REFINEMENT_SOURCE
    field.cpp
    grid_refinement.cpp
    grid_refinement_uniform.cpp
    grid_refinement_fixed_fraction.cpp
    grid_refinement_continuous.cpp
    gmsh_out.cpp
    msh_out.cpp
    gnu_out.cpp
    size_field.cpp
    reconstruct_poly.cpp
    )

# Output library
foreach(dim RANGE 1 3)
    string(CONCAT GridRefinementLib GridRefinement_${dim}D)
    add_library(${GridRefinementLib} STATIC ${GRID_REFINEMENT_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${GridRefinementLib} PRIVATE 
        PHILIP_DIM=${dim} 
        ENABLE_GMSH=${ENABLE_GMSH}
        GMSH_PATH=\"gmsh\" 
        ENABLE_GNUPLOT=${ENABLE_GNUPLOT})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${GridRefinementLib})
    endif()

    unset(GridRefinementLib)

endforeach()