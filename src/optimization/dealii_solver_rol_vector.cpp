#include <deal.II/lac/vector_memory.templates.h>

#include "optimization/dealii_solver_rol_vector.hpp"

template class dealii::VectorMemory<dealiiSolverVectorWrappingROL<double>>;
template class dealii::GrowingVectorMemory<dealiiSolverVectorWrappingROL<double>>;
