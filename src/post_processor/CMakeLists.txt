SET(SOURCE
    physics_post_processor.cpp
    )

set(POSSIBLE_N_SPECIES "0;2;5;9;11")

foreach(nspecies ${POSSIBLE_N_SPECIES})
foreach(dim RANGE 1 3)
    # Output library
    if(${nspecies} EQUAL "0")
        string(CONCAT PostprocessingLib Postprocessing_${dim}D)
    else()
        string(CONCAT PostprocessingLib Postprocessing_${dim}D_${nspecies}-SPECIES)
    endif()
    add_library(${PostprocessingLib} STATIC ${SOURCE})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${PostprocessingLib})
    endif()

    target_compile_definitions(${PostprocessingLib} PRIVATE PHILIP_DIM=${dim})
    target_compile_definitions(${PostprocessingLib} PRIVATE N_SPECIES=${nspecies})
    # target_link_libraries(${PostprocessingLib} m mpi mpi_cxx)

    # Setup target with deal.II
    unset(PostprocessingLib)

endforeach()
endforeach()
