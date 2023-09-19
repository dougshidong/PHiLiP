#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1_eulerian.h>


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include "maximum_principle_limiter.h"
#include "tvb_limiter.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
/**********************************
*
* Maximum Principle Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
MaximumPrincipleLimiter<dim, nstate, real>::MaximumPrincipleLimiter(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiter<dim,real>::BoundPreservingLimiter(nstate, parameters_input) 
{
    // Create pointer to TVB Limiter class if use_tvb_limiter==true && dim == 1
    if (parameters_input->limiter_param.use_tvb_limiter) {
        if (dim == 1) {
            tvbLimiter = std::make_shared < TVBLimiter<dim, nstate, real> >(parameters_input);
        }
        else {
            std::cout << "Error: Cannot create TVB limiter for dim > 1" << std::endl;
            std::abort();
        }
    }
}

template <int dim, int nstate, typename real>
void MaximumPrincipleLimiter<dim, nstate, real>::get_global_max_and_min_of_solution(
    dealii::LinearAlgebra::distributed::Vector<double>      solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection)
{
    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const int poly_degree = i_fele;

        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[poly_degree];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        if (this->global_max.size() < nstate && this->global_min.size() < nstate) {
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                this->global_max.push_back(-1e9);
                this->global_min.push_back(1e9);
            }
        }

        // Allocate solution dofs and set global max and min
        std::array<std::vector<real>, nstate> soln_dofs;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            if (ishape == 0) {
                soln_dofs[istate].resize(n_shape_fns);
            }
            soln_dofs[istate][ishape] = solution[current_dofs_indices[idof]]; //
            if (soln_dofs[istate][ishape] > this->global_max[istate])
                this->global_max[istate] = soln_dofs[istate][ishape];
            if (soln_dofs[istate][ishape] < this->global_min[istate])
                this->global_min[istate] = soln_dofs[istate][ishape];
        }
    }

    // Print the obtained values for verification
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        std::cout << std::fixed;
        std::cout << std::setprecision(14);
        std::cout << "global_max:   " << this->global_max[istate] << "   global_min:   " << this->global_min[istate] << std::endl;
    }
}

template <int dim, int nstate, typename real>
std::array<real, nstate> MaximumPrincipleLimiter<dim, nstate, real>::get_soln_cell_avg(
    std::array<std::vector<real>, nstate> soln_at_q,
    const unsigned int n_quad_pts,
    const std::vector<real>& quad_weights)
{
    std::array<real, nstate> soln_cell_avg;
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        soln_cell_avg[istate] = 0;
    }

    // Apply integral for solution cell average (dealii quadrature operates from [0,1])
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            soln_cell_avg[istate] += quad_weights[iquad]
                * soln_at_q[istate][iquad];
        }
    }

    return soln_cell_avg;
}

template <int dim, int nstate, typename real>
void MaximumPrincipleLimiter<dim, nstate, real>::write_limited_solution(
    dealii::LinearAlgebra::distributed::Vector<double>      solution,
    std::array<std::vector<real>, nstate>                   soln_dofs,
    const unsigned int                                      n_shape_fns,
    std::vector<dealii::types::global_dof_index>            current_dofs_indices)
{
    // Write limited solution back to global solution & verify that strict maximum principle is satisfied
    for (int istate = 0; istate < nstate; istate++) {
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            const unsigned int idof = istate * n_shape_fns + ishape;
            solution[current_dofs_indices[idof]] = soln_dofs[istate][ishape];
            if (solution[current_dofs_indices[idof]] > this->global_max[istate] + 1e-13) {
                std::cout << "Error: Solution exceeds global maximum   -   Aborting... Value:   " << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }
            if (solution[current_dofs_indices[idof]] < this->global_min[istate] - 1e-13) {
                std::cout << "Error: Solution exceeds global minimum   -   Aborting... Value:   " << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }
        }
    }
}

template <int dim, int nstate, typename real>
void MaximumPrincipleLimiter<dim, nstate, real>::limit(
    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    dealii::hp::QCollection<dim>                            volume_quadrature_collection,
    unsigned int                                            tensor_degree,
    unsigned int                                            max_degree,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
    dealii::hp::QCollection<1>                              oneD_quadrature_collection)
{
    // If use_tvb_limiter is true, apply TVB limiter before applying maximum-principle-satisfying limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, tensor_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection);

    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = tensor_degree;
    //Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim> soln_basis(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim> soln_basis_projection_oper(1, max_degree, init_grid_degree);


    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

    // Obtain the global max and min (ie. max and min of the initial solution)
    if (this->global_max.empty() && this->global_min.empty())
        get_global_max_and_min_of_solution(solution, dof_handler, fe_collection);

    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const int poly_degree = i_fele;

        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[poly_degree];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<real>, nstate> soln_dofs;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        std::array<real, nstate> local_max;
        std::array<real, nstate> local_min;
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            local_max[istate] = -1e9;
            local_min[istate] = 1e9;

            soln_dofs[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local max and min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_dofs[istate][ishape] = solution[current_dofs_indices[idof]];

            if (soln_dofs[istate][ishape] > local_max[istate])
                local_max[istate] = soln_dofs[istate][ishape];

            if (soln_dofs[istate][ishape] < local_min[istate])
                local_min[istate] = soln_dofs[istate][ishape];
        }

        const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
        const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();

        std::array<std::vector<real>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q[istate], soln_basis.oneD_vol_operator);
        }

        // Obtain solution cell average
        std::array<real, nstate> soln_cell_avg = get_soln_cell_avg(soln_at_q, n_quad_pts, quad_weights);

        // Obtain theta value
        std::array<real, nstate> theta; // Value used to linearly scale solution 
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            real maxscale = 1.0;
            real minscale = 1.0;

            if (local_max[istate] - soln_cell_avg[istate] != 0)
                maxscale = local_max[istate] - soln_cell_avg[istate];
            if (local_min[istate] - soln_cell_avg[istate] != 0)
                minscale = local_min[istate] - soln_cell_avg[istate];

            theta[istate] = std::min({ abs((this->global_max[istate] - soln_cell_avg[istate]) / maxscale),
                                        abs((this->global_min[istate] - soln_cell_avg[istate]) / minscale), 1.0 });
        }

        // Apply limiter on solution values at quadrature points
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                soln_at_q[istate][iquad] = theta[istate] * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                    + soln_cell_avg[istate];
            }
        }

        // Project solution at quadrature points to dofs.
        for (int istate = 0; istate < nstate; istate++) {
            soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_dofs[istate],
                soln_basis_projection_oper.oneD_vol_operator);
        }

        // Write limited solution back and verify that the strict maximum principle is satisfied
        write_limited_solution(solution, soln_dofs, n_shape_fns, current_dofs_indices);
    }
}

template class MaximumPrincipleLimiter <PHILIP_DIM, 1, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 2, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 3, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 4, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 5, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 6, double>;
} // PHiLiP namespace