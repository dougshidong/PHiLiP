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

set(NMPI 1)
add_test(
  NAME ${TEST_TARGET}_nmpi=${NMPI}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

if(NOT NMPI EQUAL ${MPIMAX})
  set(NMPI ${MPIMAX})
  add_test(
    NAME ${TEST_TARGET}_nmpi=${NMPI}
    COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
  )
endif()

unset(TEST_TARGET)

string(CONCAT TEST_TARGET constrained_rosenbrock)
set(TEST_SRC
    constrained_rosenbrock.cpp
    )
# Output executable
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
string(CONCAT OptimizationLib Optimization_1D)
target_link_libraries(${TEST_TARGET} ${OptimizationLib})
target_link_libraries(${TEST_TARGET} ${Boost_LIBRARIES})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

set(NMPI 1)
add_test(
  NAME ${TEST_TARGET}_nmpi=${NMPI}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set(NMPI ${MPIMAX})
add_test(
  NAME ${TEST_TARGET}_nmpi=${NMPI}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)

string(CONCAT TEST_TARGET pdas_constrained_optimization)
set(TEST_SRC
    pdas_constrained_optimization.cpp
    )
# Output executable
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
string(CONCAT OptimizationLib Optimization_1D)
target_link_libraries(${TEST_TARGET} ${OptimizationLib})
target_link_libraries(${TEST_TARGET} ${Boost_LIBRARIES})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

foreach(objectivetype RANGE 0 2)
foreach(constrainttype RANGE 0 1)
    if(${objectivetype} EQUAL 0)
        set(OBJSTRING quadratic)
    endif()
    if(${objectivetype} EQUAL 1)
        set(OBJSTRING polynomial)
    endif()
    if(${objectivetype} EQUAL 2)
        set(OBJSTRING rosenbrock)
    endif()
    if(${constrainttype} EQUAL 0)
        set(CSTSTRING linear)
    endif()
    if(${constrainttype} EQUAL 1)
        set(CSTSTRING quadratic)
    endif()
    set(NMPI ${MPIMAX})
    add_test(
      NAME ${TEST_TARGET}_${OBJSTRING}Objective_${CSTSTRING}Constraint_nmpi${NMPI}_hessian
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET} --objective_type=${objectivetype} --constraint_type=${constrainttype}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )
    add_test(
      NAME ${TEST_TARGET}_${OBJSTRING}Objective_${CSTSTRING}Constraint_nmpi${NMPI}_bfgs
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET} --objective_type=${objectivetype} --constraint_type=${constrainttype} --use_bfgs
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )
    if (NOT( ${NMPI} EQUAL 1))
        set(NMPI 1)
        add_test(
          NAME ${TEST_TARGET}_${OBJSTRING}Objective_${CSTSTRING}Constraint_nmpi${NMPI}_hessian
          COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET} --objective_type=${objectivetype} --constraint_type=${constrainttype}
          WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
        )
        add_test(
          NAME ${TEST_TARGET}_${OBJSTRING}Objective_${CSTSTRING}Constraint_nmpi${NMPI}_bfgs
          COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET} --objective_type=${objectivetype} --constraint_type=${constrainttype} --use_bfgs
          WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
        )
    endif()
endforeach()
endforeach()
unset(TEST_TARGET)


set(TEST_SRC
    flow_constraints_check.cpp
    )

set (dim 2)

# Output executable
string(CONCAT TEST_TARGET flow_constraints_check)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

target_link_libraries(${TEST_TARGET} ParametersLibrary)
target_link_libraries(${TEST_TARGET} Grids_${dim}D)
target_link_libraries(${TEST_TARGET} Physics_${dim}D)
target_link_libraries(${TEST_TARGET} InitialConditions_${dim}D)
target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
target_link_libraries(${TEST_TARGET} ODESolver_${dim}D)
target_link_libraries(${TEST_TARGET} Optimization_${dim}D)

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=2)
set(NMPI ${MPIMAX})
add_test(
  NAME ${TEST_TARGET}_nmpi=${NMPI}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)

set(TEST_SRC
    objective_check.cpp
    )

set (dim 2)

# Output executable
string(CONCAT TEST_TARGET objective_check)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

target_link_libraries(${TEST_TARGET} ParametersLibrary)
target_link_libraries(${TEST_TARGET} Grids_${dim}D)
target_link_libraries(${TEST_TARGET} Physics_${dim}D)
target_link_libraries(${TEST_TARGET} InitialConditions_${dim}D)
target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
target_link_libraries(${TEST_TARGET} ODESolver_${dim}D)
target_link_libraries(${TEST_TARGET} Functional_${dim}D)
target_link_libraries(${TEST_TARGET} Optimization_${dim}D)

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=2)
set(NMPI ${MPIMAX})
add_test(
  NAME ${TEST_TARGET}_nmpi=${NMPI}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)

set(TEST_SRC
    preconditioner_eigenvalues.cpp
    )

set (dim 2)

# Output executable
string(CONCAT TEST_TARGET birosghattas_preconditioned_system_eigenvalues)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

target_link_libraries(${TEST_TARGET} ParametersLibrary)
target_link_libraries(${TEST_TARGET} Grids_${dim}D)
target_link_libraries(${TEST_TARGET} Physics_${dim}D)
target_link_libraries(${TEST_TARGET} InitialConditions_${dim}D)
target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
target_link_libraries(${TEST_TARGET} ODESolver_${dim}D)
target_link_libraries(${TEST_TARGET} Functional_${dim}D)
target_link_libraries(${TEST_TARGET} Optimization_${dim}D)

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=2)
set(NMPI ${MPIMAX})
add_test(
  NAME ${TEST_TARGET}
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)
