#include "maximum_principle_limiter.h"
#include "tvb_limiter.h"

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
    : BoundPreservingLimiterState<dim,nstate, real>::BoundPreservingLimiterState(parameters_input) 
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
    const dealii::LinearAlgebra::distributed::Vector<double>&   solution,
    const dealii::DoFHandler<dim>&                              dof_handler,
    const dealii::hp::FECollection<dim>&                        fe_collection)
{
    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        if (global_max.size() < nstate && global_min.size() < nstate) {
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                global_max.push_back(-1e9);
                global_min.push_back(1e9);
            }
        }

        // Allocate solution dofs and set global max and min
        std::array<std::vector<real>, nstate> soln_coeff;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            if (ishape == 0) {
                soln_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]]; //
            if (soln_coeff[istate][ishape] > global_max[istate])
                global_max[istate] = soln_coeff[istate][ishape];
            if (soln_coeff[istate][ishape] < global_min[istate])
                global_min[istate] = soln_coeff[istate][ishape];
        }
    }

    // Print the obtained values for verification
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        std::cout << std::fixed;
        std::cout << std::setprecision(14);
        std::cout << "global_max:   " << global_max[istate] << "   global_min:   " << global_min[istate] << std::endl;
    }
}

template <int dim, int nstate, typename real>
void MaximumPrincipleLimiter<dim, nstate, real>::write_limited_solution(
    dealii::LinearAlgebra::distributed::Vector<double>&      solution,
    const std::array<std::vector<real>, nstate>&             soln_coeff,
    const unsigned int                                       n_shape_fns,
    const std::vector<dealii::types::global_dof_index>&      current_dofs_indices)
{
    // Write limited solution back to global solution & verify that strict maximum principle is satisfied
    for (int istate = 0; istate < nstate; istate++) {
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            const unsigned int idof = istate * n_shape_fns + ishape;
            solution[current_dofs_indices[idof]] = soln_coeff[istate][ishape];

            if (solution[current_dofs_indices[idof]] > global_max[istate] + 1e-13) {
                std::cout << "Error: Solution exceeds global maximum   -   Aborting... Value:   " << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }
            if (solution[current_dofs_indices[idof]] < global_min[istate] - 1e-13) {
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
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
        double                                                  dt)
{
    // If use_tvb_limiter is true, apply TVB limiter before applying maximum-principle-satisfying limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection, dt);

    // Construct 1D Quad Points
    const unsigned int init_grid_degree = grid_degree;
    dealii::QGauss<1> oneD_quad_GL(max_degree + 1);
    dealii::QGaussLobatto<1> oneD_quad_GLL(max_degree + 1);

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_GLL(1, max_degree, init_grid_degree);
    soln_basis_GLL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quad_GLL);
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_GL(1, max_degree, init_grid_degree);
    soln_basis_GL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quad_GL);

    // Obtain the global max and min (ie. max and min of the initial solution)
    if (global_max.empty() && global_min.empty())
        get_global_max_and_min_of_solution(solution, dof_handler, fe_collection);

    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<real>, nstate> soln_coeff;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        std::array<real, nstate> local_max;
        std::array<real, nstate> local_min;
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            local_max[istate] = -1e9;
            local_min[istate] = 1e9;

            soln_coeff[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local max and min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]];

            if (soln_coeff[istate][ishape] > local_max[istate])
                local_max[istate] = soln_coeff[istate][ishape];

            if (soln_coeff[istate][ishape] < local_min[istate])
                local_min[istate] = soln_coeff[istate][ishape];
        }


        const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
        const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();

        std::array<std::array<std::vector<real>, nstate>, dim> soln_at_q;
        std::array<std::vector<real>, nstate> soln_at_q_dim;

        // Interpolate solution dofs to quadrature pts.
        for(unsigned int idim = 0; idim < dim; idim++) {
            for (int istate = 0; istate < nstate; istate++) {
                soln_at_q_dim[istate].resize(n_quad_pts);

                if(idim == 0) {
                    soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                        soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
                }

                if(idim == 1) {
                    soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                        soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
                }

                if(idim == 2) {
                    soln_basis_GLL.matrix_vector_mult(soln_coeff[istate], soln_at_q_dim[istate],
                        soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator);
                }
            }
            soln_at_q[idim] = soln_at_q_dim;
        }

        for (unsigned int idim = 0; idim < dim; ++idim) {
             for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    if(soln_at_q[idim][istate][iquad] > local_max[istate]) {
                        local_max[istate] = soln_at_q[idim][istate][iquad];
                    }
                    if(soln_at_q[idim][istate][iquad] < local_min[istate]) {
                        local_min[istate] = soln_at_q[idim][istate][iquad];
                    }
                }
            }
        }

        // Obtain solution cell average
        std::array<real, nstate> soln_cell_avg = get_soln_cell_avg(soln_coeff, n_quad_pts, quad_weights);

        // Obtain theta value
        std::array<real, nstate> theta; // Value used to linearly scale solution 
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            real maxscale = 1.0;
            real minscale = 1.0;

            if (local_max[istate] - soln_cell_avg[istate] != 0)
                maxscale = local_max[istate] - soln_cell_avg[istate];
            if (local_min[istate] - soln_cell_avg[istate] != 0)
                minscale = local_min[istate] - soln_cell_avg[istate];

            theta[istate] = std::min({ abs((global_max[istate] - soln_cell_avg[istate]) / maxscale),
                                        abs((global_min[istate] - soln_cell_avg[istate]) / minscale), 1.0 });
        }

        // Apply limiter on solution values at quadrature points
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                soln_coeff[istate][iquad] = theta[istate] * (soln_coeff[istate][iquad] - soln_cell_avg[istate])
                    + soln_cell_avg[istate];
            }
        }

        // Write limited solution back and verify that the strict maximum principle is satisfied
        write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);
    }
}

template class MaximumPrincipleLimiter <PHILIP_DIM, 1, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 2, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 3, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 4, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 5, double>;
template class MaximumPrincipleLimiter <PHILIP_DIM, 6, double>;
} // PHiLiP namespace