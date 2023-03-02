set(MESH_ADAPTATION_SRC
	mesh_error_factory.cpp
	mesh_error_estimate.cpp
	mesh_adaptation.cpp)

foreach(dim RANGE 1 3)

	string(CONCAT MeshAdaptationLib MeshAdaptation_${dim}D)
	add_library(${MeshAdaptationLib} STATIC ${MESH_ADAPTATION_SRC})

    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${MeshAdaptationLib} PRIVATE PHILIP_DIM=${dim})

	# Link with HighOrderGridLib
	string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
	target_link_libraries(${MeshAdaptationLib} ${HighOrderGridLib})

	# Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${MeshAdaptationLib})
    endif()

	unset(HighOrderGridLib)
	unset(MeshAdaptationLib)
endforeach()
