set(DESIGN_PARAMETERIZATION_SRC
    base_parameterization.cpp
    ffd_parameterization.cpp
    identity_parameterization.cpp
    inner_vol_parameterization.cpp
    )

foreach(dim RANGE 1 3)

    string(CONCAT DesignParameterizationLib DesignParameterization_${dim}D)
    add_library(${DesignParameterizationLib} STATIC ${DESIGN_PARAMETERIZATION_SRC})
    
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${DesignParameterizationLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT GridsLib Grids_${dim}D)
    target_link_libraries(${DesignParameterizationLib} ${GridsLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${DesignParameterizationLib})
    endif()

    unset(DesignParameterizationLib)
    unset(GridsLib)

endforeach()


    
