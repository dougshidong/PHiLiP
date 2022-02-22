# Set of existing grids that are generated through deal.II

set(GRIDS_SOURCE
    gaussian_bump.cpp
    curved_periodic_grid.cpp
    wavy_periodic_grid.cpp
    naca_airfoil_grid.cpp
    spline_channel.cpp
    straight_periodic_cube.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT GridsLib Grids_${dim}D)
    add_library(${GridsLib} STATIC ${GRIDS_SOURCE})
    # Link with HighOrderGridLib
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    target_link_libraries(${GridsLib} ${HighOrderGridLib})

    target_compile_definitions(${GridsLib} PRIVATE PHILIP_DIM=${dim})
    unset(HighOrderGridLib)

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${GridsLib})
    endif()


    unset(GridsLib)
endforeach()
