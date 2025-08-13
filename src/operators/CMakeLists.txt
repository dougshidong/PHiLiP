SET(OPERSOURCE
    operators.cpp
#    operators_factory.cpp
    )

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # Output library
    if(species MATCHES 1)
        string(CONCAT OperatorsLib Operator_Lib_${dim}D)
    else()
        string(CONCAT OperatorsLib Operator_Lib_${dim}D_${species}species)
    endif()

    add_library(${OperatorsLib} STATIC ${OPERSOURCE})

    target_compile_definitions(${OperatorsLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})
    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${ParameterLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${OperatorsLib})
    endif()

    unset(OperatorsLib)

endforeach()
endforeach()
