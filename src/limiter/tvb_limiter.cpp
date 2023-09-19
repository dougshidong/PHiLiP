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

#include "tvb_limiter.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
/**********************************
*
* TVB Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
TVBLimiter<dim, nstate, real>::TVBLimiter(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiter<dim,real>::BoundPreservingLimiter(nstate, parameters_input) {}

template <int dim, int nstate, typename real>
std::array<std::vector<real>, nstate> TVBLimiter<dim, nstate, real>::limit_cell(
    std::array<std::vector<real>, nstate>                   soln_at_q,
    const unsigned int                                      n_quad_pts,
    std::array<real, nstate>                                prev_cell_avg,
    std::array<real, nstate>                                soln_cell_avg,
    std::array<real, nstate>                                next_cell_avg,
    std::array<real, nstate>                                M,
    double                                                  h)
{
    std::array<real, nstate> soln_cell_0;
    std::array<real, nstate> soln_cell_k;
    std::array<real, nstate> diff_next;
    std::array<real, nstate> diff_prev;
    std::array<real, nstate> soln_0_lim;
    std::array<real, nstate> soln_k_lim;
    std::array<real, nstate> theta; // Value used to linearly scale solution 

    // This part is only valid for 1D as it uses the value at the first and last quadrature points 
    // as the face values of the cell which does not hold true for 2D and 3D.
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        soln_cell_0[istate] = soln_at_q[istate][0]; // Value at left face of the cell
        soln_cell_k[istate] = soln_at_q[istate][n_quad_pts - 1]; // Value at right face of the cell

        diff_next[istate] = next_cell_avg[istate] - soln_cell_avg[istate];
        diff_prev[istate] = soln_cell_avg[istate] - prev_cell_avg[istate];
    }

    real a = 0.0; // Chen,Shu 2017, Thm 3.7 minmod function
    real minmod = 0.0;

    for (unsigned int istate = 0; istate < nstate; ++istate) {
        a = soln_cell_avg[istate] - soln_cell_0[istate];
        if (abs(a) <= M[istate] * pow(h, 2.0)) {
            soln_0_lim[istate] = soln_cell_avg[istate] - a;
        }
        else {
            if (signbit(a) == signbit(diff_next[istate]) && signbit(a) == signbit(diff_prev[istate])) {
                minmod = std::min({ abs(a), abs(diff_next[istate]), abs(diff_prev[istate]) });

                if (signbit(a))
                    soln_0_lim[istate] = soln_cell_avg[istate] + std::max(abs(minmod), M[istate] * pow(h, 2.0));
                else
                    soln_0_lim[istate] = soln_cell_avg[istate] - std::max(abs(minmod), M[istate] * pow(h, 2.0));
            }
            else
                soln_0_lim[istate] = soln_cell_avg[istate] - M[istate] * pow(h, 2.0);
        }

        a = soln_cell_k[istate] - soln_cell_avg[istate];
        if (abs(a) <= M[istate] * pow(h, 2.0)) {
            soln_k_lim[istate] = soln_cell_avg[istate] + a;
        }
        else {
            if (signbit(a) == signbit(diff_next[istate]) && signbit(a) == signbit(diff_prev[istate])) {
                minmod = std::min({ abs(a), abs(diff_next[istate]), abs(diff_prev[istate]) });

                if (signbit(a))
                    soln_k_lim[istate] = soln_cell_avg[istate] - std::max(abs(minmod), M[istate] * pow(h, 2.0));
                else
                    soln_k_lim[istate] = soln_cell_avg[istate] + std::max(abs(minmod), M[istate] * pow(h, 2.0));
            }
            else
                soln_k_lim[istate] = soln_cell_avg[istate] + M[istate] * pow(h, 2.0);
        }

        real scale = ((soln_cell_0[istate] - soln_cell_avg[istate]) + (soln_cell_k[istate] - soln_cell_avg[istate]));
        if (scale != 0)
            theta[istate] = ((soln_0_lim[istate] - soln_cell_avg[istate]) + (soln_k_lim[istate] - soln_cell_avg[istate])) / scale;
        else
            theta[istate] = 0;
    }

    // Limit the values at the quadrature points
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            if (iquad == 0) {
                soln_at_q[istate][iquad] = soln_0_lim[istate];
            }
            else if (iquad == n_quad_pts - 1) {
                soln_at_q[istate][iquad] = soln_k_lim[istate];
            }
            else {
                soln_at_q[istate][iquad] = theta[istate] * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                    + soln_cell_avg[istate];
            }
        }
    }

    return soln_at_q;
}

template <int dim, int nstate, typename real>
std::array<real, nstate> TVBLimiter<dim, nstate, real>::get_neighbour_cell_avg(
    dealii::LinearAlgebra::distributed::Vector<double>      solution,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    dealii::hp::QCollection<dim>                            volume_quadrature_collection,
    OPERATOR::basis_functions<dim, 2 * dim>                 soln_basis,
    const int                                               poly_degree,
    std::vector<dealii::types::global_dof_index>            neigh_dofs_indices,
    const unsigned int                                      n_dofs_neigh_cell)
{
    // Extract the local solution dofs in the cell from the global solution dofs
    std::array<std::vector<real>, nstate> soln_dofs;
    const unsigned int n_shape_fns = n_dofs_neigh_cell / nstate;

    for (unsigned int istate = 0; istate < nstate; ++istate) {
        soln_dofs[istate].resize(n_shape_fns);
    }

    // Allocate soln_dofs
    for (unsigned int idof = 0; idof < n_dofs_neigh_cell; ++idof) {
        const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
        soln_dofs[istate][ishape] = solution[neigh_dofs_indices[idof]];
    }
    const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
    const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();

    std::array<real, nstate> cell_avg;
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        cell_avg[istate] = 0;
    }

    std::array<std::vector<real>, nstate> soln_at_q_neigh;

    // Interpolate solution dofs to quadrature pts
    for (int istate = 0; istate < nstate; istate++) {
        soln_at_q_neigh[istate].resize(n_quad_pts);
        soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q_neigh[istate],
            soln_basis.oneD_vol_operator);
    }

    // Apply integral for solution cell average (dealii quadrature operates from [0,1])
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            cell_avg[istate] += quad_weights[iquad]
                * soln_at_q_neigh[istate][iquad];
        }
    }

    return cell_avg;
}

template <int dim, int nstate, typename real>
std::array<real, nstate> TVBLimiter<dim, nstate, real>::get_current_cell_avg(
    std::array<std::vector<real>, nstate>                   soln_at_q,
    const unsigned int                                      n_quad_pts,
    const std::vector<real>&                                quad_weights)
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
void TVBLimiter<dim, nstate, real>::limit(
    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    dealii::hp::QCollection<dim>                            volume_quadrature_collection,
    unsigned int                                            tensor_degree,
    unsigned int                                            max_degree,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
    dealii::hp::QCollection<1>                              oneD_quadrature_collection)
{
    double h = this->all_parameters->limiter_param.tvb_h;

    std::array<real, nstate> M;
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        M[istate] = this->all_parameters->limiter_param.tvb_M[istate];
    }

    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = tensor_degree;
    //Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim> soln_basis(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim> soln_basis_projection_oper(1, max_degree, init_grid_degree);


    //build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::array<real, nstate> prev_cell_avg;
        std::array<real, nstate> next_cell_avg;

        // Initialize all cell_avg arrays to zero
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            prev_cell_avg[istate] = 0;
            next_cell_avg[istate] = 0;
        }

        for (const auto face_no : soln_cell->face_indices()) {
            if (soln_cell->neighbor(face_no).state() != dealii::IteratorState::valid) continue;

            std::vector<dealii::types::global_dof_index> neigh_dofs_indices;
            // Current reference element related to this physical cell
            auto neigh = soln_cell->neighbor(face_no);
            const int i_fele = neigh->active_fe_index();
            const int poly_degree = i_fele;

            const dealii::FESystem<dim, dim>& neigh_fe_ref = fe_collection[poly_degree];
            const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
            // Obtain the mapping from local dof indices to global dof indices
            neigh_dofs_indices.resize(n_dofs_neigh_cell);
            neigh->get_dof_indices(neigh_dofs_indices);
            
            std::array<real, nstate> neigh_cell_avg = get_neighbour_cell_avg(solution, fe_collection, volume_quadrature_collection, soln_basis,
                poly_degree, neigh_dofs_indices, n_dofs_neigh_cell);

            if (face_no == 0) {
                prev_cell_avg = neigh_cell_avg;
            }
            else if (face_no == 1) {
                next_cell_avg = neigh_cell_avg;
            }
        }

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

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_dofs[istate].resize(n_shape_fns);
        }

        // Allocate soln_dofs
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_dofs[istate][ishape] = solution[current_dofs_indices[idof]];
        }


        const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
        const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();

        std::array<std::vector<real>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q[istate],
                soln_basis.oneD_vol_operator);
        }

        std::array<real, nstate> soln_cell_avg = get_current_cell_avg(soln_at_q, n_quad_pts, quad_weights);

        std::array<std::vector<real>, nstate> soln_at_q_lim = limit_cell(soln_at_q, n_quad_pts, prev_cell_avg, soln_cell_avg, next_cell_avg, M, h);

        // Project solution at quadrature points to dofs.
        for (int istate = 0; istate < nstate; istate++) {
            soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q_lim[istate], soln_dofs[istate],
                soln_basis_projection_oper.oneD_vol_operator);
        }

        // Write limited solution dofs to the global solution vector.
        for (int istate = 0; istate < nstate; istate++) {
            for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                const unsigned int idof = istate * n_shape_fns + ishape;
                solution[current_dofs_indices[idof]] = soln_dofs[istate][ishape];
            }
        }
    }
}

template class TVBLimiter <PHILIP_DIM, 1, double>;
template class TVBLimiter <PHILIP_DIM, 2, double>;
template class TVBLimiter <PHILIP_DIM, 3, double>;
template class TVBLimiter <PHILIP_DIM, 4, double>;
template class TVBLimiter <PHILIP_DIM, 5, double>;
template class TVBLimiter <PHILIP_DIM, 6, double>;
} // PHiLiP namespace