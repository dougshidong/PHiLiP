#ifndef PHILIP_LOCAL_SOLUTION_HPP
#define PHILIP_LOCAL_SOLUTION_HPP

#include <deal.II/fe/fe_system.h>

namespace PHiLiP {

/// Class to store local solution coefficients and provide evaluation functions.
/** This class is used to store the solution coefficients in the finite element basis and provide functions to evaluate the solution
 * value and gradients at arbitrary points. It can be used for both state and metric solutions since they are both represented by a
 * finite element discretization.
 * The template parameters are: \n
 * - real: The floating point type used to represent the solution coefficients. This is usually double or an AD type.
 * - dim: The dimension of the problem.
 * - n_components: The number of components of the solution. This is nstate for state solutions and dim for metric solutions.
 */
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
