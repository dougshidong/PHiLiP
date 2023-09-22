#ifndef PHILIP_LOCAL_SOLUTION_HPP
#define PHILIP_LOCAL_SOLUTION_HPP

#include <deal.II/fe/fe_system.h>

#include <vector>

template <typename real, int dim, int n_components>
struct LocalSolution {
    std::vector<real> coefficients;
    const dealii::FESystem<dim, dim> &finite_element;
    const unsigned int n_dofs;

    LocalSolution(const dealii::FESystem<dim, dim> &finite_element)
        : finite_element(finite_element), n_dofs(finite_element.dofs_per_cell) {
        Assert(n_components == finite_element.n_components(),
               dealii::ExcMessage("Number of components must match finite element system"));
        coefficients.resize(finite_element.dofs_per_cell);
    }

    std::vector<std::array<real, n_components>> evaluate_values(
        const std::vector<dealii::Point<dim>> &unit_points) const {
        const unsigned int n_points = unit_points.size();
        std::vector<std::array<real, n_components>> values(n_points);

        for (unsigned int i_point = 0; i_point < n_points; ++i_point) {
            for (int i_component = 0; i_component < n_components; ++i_component) {
                values[i_point][i_component] = 0;
            }
            for (unsigned int i_dof = 0; i_dof < n_dofs; ++i_dof) {
                const int i_component = finite_element.system_to_component_index(i_dof).first;
                values[i_point][i_component] += coefficients[i_dof] * finite_element.shape_value_component(
                                                                          i_dof, unit_points[i_point], i_component);
            }
        }

        return values;
    }

    std::vector<std::array<dealii::Tensor<1, dim, real>, n_components>> evaluate_reference_gradients(
        const std::vector<dealii::Point<dim>> &unit_points) const {
        const unsigned int n_points = unit_points.size();
        std::vector<std::array<dealii::Tensor<1, dim, real>, n_components>> gradients(unit_points.size());

        for (unsigned int i_point = 0; i_point < n_points; ++i_point) {
            for (int i_component = 0; i_component < n_components; ++i_component) {
                gradients[i_point][i_component] = 0;
            }
            for (unsigned int i_dof = 0; i_dof < n_dofs; ++i_dof) {
                const int i_component = finite_element.system_to_component_index(i_dof).first;
                dealii::Tensor<1, dim, double> shape_grad =
                    finite_element.shape_grad_component(i_dof, unit_points[i_point], i_component);
                for (int d = 0; d < dim; ++d) {
                    gradients[i_point][i_component][d] += coefficients[i_dof] * shape_grad[d];
                }
            }
        }
        return gradients;
    }
};

#endif  // PHILIP_LOCAL_SOLUTION_HPP
