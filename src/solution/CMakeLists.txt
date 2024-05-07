set(SOURCE
        local_solution.cpp
)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT SolutionLib Solution_${dim}D)
    add_library(${SolutionLib} STATIC ${SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${SolutionLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${SolutionLib})
    endif()

endforeach()
