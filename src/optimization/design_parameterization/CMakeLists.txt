set(DESIGN_PARAMETERIZATION_SRC
    base_parameterization.cpp
    ffd_parameterization.cpp
    identity_parameterization.cpp
    inner_vol_parameterization.cpp
    )

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    string(CONCAT DesignParameterizationLib DesignParameterization ${lib_spec})
    add_library(${DesignParameterizationLib} STATIC ${DESIGN_PARAMETERIZATION_SRC})
    
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${DesignParameterizationLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1, 2, or 3 in the code
    target_compile_definitions(${DesignParameterizationLib} PRIVATE PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT GridsLib Grids ${lib_spec})
    target_link_libraries(${DesignParameterizationLib} ${GridsLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${DesignParameterizationLib})
    endif()

    unset(DesignParameterizationLib)
    unset(GridsLib)
endforeach()
endforeach()


    
