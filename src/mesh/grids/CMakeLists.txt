# Set of existing grids that are generated through deal.II

set(GRIDS_SOURCE
    gaussian_bump.cpp
    curved_periodic_grid.cpp
    wavy_periodic_grid.cpp
    naca_airfoil_grid.cpp
    spline_channel.cpp
    straight_periodic_cube.cpp
    nonsymmetric_curved_periodic_grid.cpp
    nonsymmetric_curved_nonperiodic_grid.cpp
    skew_symmetric_periodic_grid.cpp
    non_periodic_cube.cpp
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
    string(CONCAT GridsLib Grids ${lib_spec})
    add_library(${GridsLib} STATIC ${GRIDS_SOURCE})
    # Link with HighOrderGridLib
    string(CONCAT HighOrderGridLib HighOrderGrid ${lib_spec})
    target_link_libraries(${GridsLib} ${HighOrderGridLib})

    target_compile_definitions(${GridsLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})
    unset(HighOrderGridLib)

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${GridsLib})
    endif()


    unset(GridsLib)
endforeach()
endforeach()
