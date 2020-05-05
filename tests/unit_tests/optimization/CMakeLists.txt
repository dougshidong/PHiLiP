set(TEST_SRC
    rol_rosenbrock.cpp
    )
# Output executable
string(CONCAT TEST_TARGET rol_rosenbrock)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

foreach(dim RANGE 1 2)

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        set(NMPI ${MPIMAX})
    endif()
    add_test(
      NAME ${TEST_TARGET}_nmpi=${NMPI}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )
endforeach()
unset(TEST_TARGET)
