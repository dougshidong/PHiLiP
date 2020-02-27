# Set of existing grids that are generated through deal.II

set(GRIDS_SOURCE
    gaussian_bump.cpp
    )

# Output library
string(CONCAT GridsLib Grids)
add_library(${GridsLib} STATIC ${GRIDS_SOURCE})
# Link with HighOrderGridLib
foreach(dim RANGE 1 3)
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    target_link_libraries(${GridsLib} ${HighOrderGridLib})
    unset(HighOrderGridLib)
endforeach()

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${GridsLib})
endif()

unset(GridsLib)
