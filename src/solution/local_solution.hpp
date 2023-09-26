#ifndef PHILIP_LOCAL_SOLUTION_HPP
#define PHILIP_LOCAL_SOLUTION_HPP

#include <deal.II/fe/fe_system.h>

#include <vector>

namespace PHiLiP {

template <typename real, int dim, int n_components>
class LocalSolution {
   public:
    /// Solution coefficients in the finite element basis.
    std::vector<real> coefficients;
    /// Reference to the finite element system used to represent the solution.
    const dealii::FESystem<dim, dim> &finite_element;
    /// Number of degrees of freedom.
    const unsigned int n_dofs;

    /// Constructor
    LocalSolution(const dealii::FESystem<dim, dim> &finite_element);

    /// Obtain values at unit points.
    std::vector<std::array<real, n_components>> evaluate_values(const std::vector<dealii::Point<dim>> &unit_points) const;

    /// Obtain reference gradients at unit points.
    /// Note that the gradients are not physical since they do not include metric terms.
    std::vector<std::array<dealii::Tensor<1, dim, real>, n_components>> evaluate_reference_gradients(
        const std::vector<dealii::Point<dim>> &unit_points) const;
};

}  // namespace PHiLiP
#endif  // PHILIP_LOCAL_SOLUTION_HPP
