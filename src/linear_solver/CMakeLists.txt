set(SOURCE
    linear_solver.cpp
    )

# Output library
string(CONCAT LinearSolverLib LinearSolver)
add_library(${LinearSolverLib} STATIC ${SOURCE})
# Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
target_compile_definitions(${LinearSolverLib} PRIVATE PHILIP_DIM=${dim})

# Setup target with deal.II
if(NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${LinearSolverLib})
endif()

unset(LinearSolverLib)

