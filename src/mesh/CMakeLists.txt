add_subdirectory(grids)
add_subdirectory(mesh_adaptation)

set(GRID_SOURCE
    high_order_grid.cpp
    gmsh_reader.cpp
    meshmover_linear_elasticity.cpp
    free_form_deformation.cpp)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT HighOrderGridLib HighOrderGrid ${lib_spec})
    add_library(${HighOrderGridLib} STATIC ${GRID_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${HighOrderGridLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT FunctionalLib Functional ${lib_spec})
    target_link_libraries(${HighOrderGridLib} ${FunctionalLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${HighOrderGridLib})
    endif()

    unset(HighOrderGridLib)
    unset(FunctionalLib)

endforeach()
endforeach()
