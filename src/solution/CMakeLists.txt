set(SOURCE
        local_solution.cpp
)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # Output library
    if(species MATCHES 1)
        string(CONCAT SolutionLib Solution_${dim}D)
    else()
        string(CONCAT SolutionLib Solution_${dim}D_${species}species)
    endif()

    add_library(${SolutionLib} STATIC ${SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${SolutionLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1, 2, or 3 in the code
    target_compile_definitions(${SolutionLib} PRIVATE PHILIP_SPECIES=${species})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${SolutionLib})
    endif()

endforeach()
endforeach()
