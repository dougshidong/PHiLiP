#ifndef PHILIP_LOCAL_SOLUTION_HPP
#define PHILIP_LOCAL_SOLUTION_HPP

#include <deal.II/fe/fe_system.h>

#include <vector>

template<typename real, int dim>
struct LocalSolution {
    LocalSolution(const dealii::FESystem<dim,dim> &finite_element)
        : finite_element(finite_element)
    {
        coefficients.resize(finite_element.dofs_per_cell);
    }
    std::vector<real> coefficients;
    const dealii::FESystem<dim,dim> &finite_element;
};

#endif  // PHILIP_LOCAL_SOLUTION_HPP
