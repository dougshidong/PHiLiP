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

#include "bound_preserving_limiter.h"

namespace PHiLiP {
    namespace LIMITER {
        // Constructor
        template <int dim, int nstate, typename real>
        BoundPreservingLimiter<dim, nstate, real>::BoundPreservingLimiter(
            const Parameters::AllParameters* const parameters_input)
            : all_parameters(parameters_input) {}

        /**********************************
        *
        * TVB Limiter Class
        *
        **********************************/
        // Constructor
        template <int dim, int nstate, typename real>
        TVBLimiter<dim, nstate, real>::TVBLimiter(
            const Parameters::AllParameters* const parameters_input)
            : all_parameters(parameters_input) {}

        template <int dim, int nstate, typename real>
        void TVBLimiter<dim, nstate, real>::limit(
            dealii::LinearAlgebra::distributed::Vector<double>      solution,
            dealii::DoFHandler<dim>                                 dof_handler,
            dealii::FESystem<dim, dim>& current_fe_ref,
            dealii::hp::QCollection<dim>                            volume_quadrature_collection)
        {
            std::array<real, nstate> M;
            double h = all_parameters->tvb_h;
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                M[istate] = 0.05 / pow(h, 2);
            }

            //create 1D solution polynomial basis functions and corresponding projection operator
            //to interpolate the solution to the quadrature nodes, and to project it back to the
            //modal coefficients.
            const unsigned int init_grid_degree = this->high_order_grid->fe_system.tensor_degree();
            //Constructor for the operators
            OPERATOR::basis_functions<dim, 2 * dim> soln_basis(1, this->max_degree, init_grid_degree);
            OPERATOR::vol_projection_operator<dim, 2 * dim> soln_basis_projection_oper(1, this->max_degree, init_grid_degree);


            //build the oneD operator to perform interpolation/projection
            soln_basis.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);
            soln_basis_projection_oper.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);

            for (auto soln_cell : this->dof_handler.active_cell_iterators()) {
                if (!soln_cell->is_locally_owned()) continue;

                std::array<real, nstate> prev_cell_avg;
                std::array<real, nstate> soln_cell_avg;
                std::array<real, nstate> next_cell_avg;

                for (const auto face_no : soln_cell->face_indices()) {
                    if (soln_cell->neighbor(face_no).state() != dealii::IteratorState::valid) continue;

                    std::vector<dealii::types::global_dof_index> neigh_dofs_indices;
                    // Current reference element related to this physical cell
                    auto neigh = soln_cell->neighbor(face_no);
                    const int i_fele = neigh->active_fe_index();
                    const int poly_degree = i_fele;

                    const dealii::FESystem<dim, dim>& neigh_fe_ref = this->fe_collection[poly_degree];
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
                    // Obtain the mapping from local dof indices to global dof indices
                    neigh_dofs_indices.resize(n_dofs_neigh_cell);
                    neigh->get_dof_indices(neigh_dofs_indices);

                    //Extract the local solution dofs in the cell from the global solution dofs
                    std::array<std::vector<real>, nstate> soln_dofs;
                    const unsigned int n_shape_fns = n_dofs_neigh_cell / nstate;

                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        //allocate soln_dofs
                        soln_dofs[istate].resize(n_shape_fns);
                    }

                    for (unsigned int idof = 0; idof < n_dofs_neigh_cell; ++idof) {
                        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
                        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
                        soln_dofs[istate][ishape] = this->solution[neigh_dofs_indices[idof]]; //
                    }
                    const unsigned int n_quad_pts = this->volume_quadrature_collection[poly_degree].size();
                    const std::vector<real>& quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();

                    //interpolate solution dofs to quadrature pts.
                    //and apply integral for the soln avg
                    std::array<real, nstate> neigh_cell_avg;

                    std::array<std::vector<real>, nstate> soln_at_q_neigh;

                    for (int istate = 0; istate < nstate; istate++) {
                        soln_at_q_neigh[istate].resize(n_quad_pts);
                        soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q_neigh[istate],
                            soln_basis.oneD_vol_operator);
                    }

                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                            neigh_cell_avg[istate] += quad_weights[iquad]
                                * soln_at_q_neigh[istate][iquad];
                        }
                    }

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

                const dealii::FESystem<dim, dim>& current_fe_ref = this->fe_collection[poly_degree];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                soln_cell->get_dof_indices(current_dofs_indices);

                //Extract the local solution dofs in the cell from the global solution dofs
                std::array<std::vector<real>, nstate> soln_dofs;
                const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    //allocate soln_dofs
                    soln_dofs[istate].resize(n_shape_fns);
                }

                for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                    const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
                    const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
                    soln_dofs[istate][ishape] = this->solution[current_dofs_indices[idof]]; //
                }


                const unsigned int n_quad_pts = this->volume_quadrature_collection[poly_degree].size();
                const std::vector<real>& quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();

                //interpolate solution dofs to quadrature pts.
                //and apply integral for the soln avg
                std::array<std::vector<real>, nstate> soln_at_q;

                for (int istate = 0; istate < nstate; istate++) {
                    soln_at_q[istate].resize(n_quad_pts);
                    soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q[istate],
                        soln_basis.oneD_vol_operator);
                }

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        soln_cell_avg[istate] += quad_weights[iquad]
                            * soln_at_q[istate][iquad];
                    }
                }

                std::array<real, nstate> soln_cell_0;
                std::array<real, nstate> soln_cell_k;
                std::array<real, nstate> diff_next;
                std::array<real, nstate> diff_prev;
                std::array<real, nstate> soln_0_lim;
                std::array<real, nstate> soln_k_lim;
                std::array<real, nstate> theta;

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    soln_cell_0[istate] = soln_at_q[istate][0];
                    soln_cell_k[istate] = soln_at_q[istate][n_quad_pts - 1];

                    diff_next[istate] = next_cell_avg[istate] - soln_cell_avg[istate];
                    diff_prev[istate] = soln_cell_avg[istate] - prev_cell_avg[istate];
                }

                real a = 0.0;
                real minmod = 0.0;

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    a = soln_cell_avg[istate] - soln_cell_0[istate];
                    if (abs(a) <= M[istate] * pow(this->h, 2.0)) {
                        soln_0_lim[istate] = soln_cell_avg[istate] - a;
                    }
                    else {
                        if (signbit(a) == signbit(diff_next[istate]) && signbit(a) == signbit(diff_prev[istate])) {
                            minmod = std::min({ abs(a), abs(diff_next[istate]), abs(diff_prev[istate]) });

                            if (signbit(a))
                                soln_0_lim[istate] = soln_cell_avg[istate] + std::max(abs(minmod), M[istate] * pow(this->h, 2.0));
                            else
                                soln_0_lim[istate] = soln_cell_avg[istate] - std::max(abs(minmod), M[istate] * pow(this->h, 2.0));
                        }
                        else
                            soln_0_lim[istate] = soln_cell_avg[istate] - M[istate] * pow(this->h, 2.0);
                    }

                    a = soln_cell_k[istate] - soln_cell_avg[istate];
                    if (abs(a) <= M[istate] * pow(this->h, 2.0)) {
                        soln_k_lim[istate] = soln_cell_avg[istate] + a;
                    }
                    else {
                        if (signbit(a) == signbit(diff_next[istate]) && signbit(a) == signbit(diff_prev[istate])) {
                            minmod = std::min({ abs(a), abs(diff_next[istate]), abs(diff_prev[istate]) });

                            if (signbit(a))
                                soln_k_lim[istate] = soln_cell_avg[istate] - std::max(abs(minmod), M[istate] * pow(this->h, 2.0));
                            else
                                soln_k_lim[istate] = soln_cell_avg[istate] + std::max(abs(minmod), M[istate] * pow(this->h, 2.0));
                        }
                        else
                            soln_k_lim[istate] = soln_cell_avg[istate] + M[istate] * pow(this->h, 2.0);
                    }

                    real scale = ((soln_cell_0[istate] - soln_cell_avg[istate]) + (soln_cell_k[istate] - soln_cell_avg[istate]));
                    if (scale != 0)
                        theta[istate] = ((soln_0_lim[istate] - soln_cell_avg[istate]) + (soln_k_lim[istate] - soln_cell_avg[istate])) / scale;
                    else
                        theta[istate] = 0;
                }

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        if (iquad == 0)
                            soln_at_q[istate][iquad] = soln_0_lim[istate];
                        else if (iquad == n_quad_pts - 1)
                            soln_at_q[istate][iquad] = soln_k_lim[istate];
                        else {
                            soln_at_q[istate][iquad] = theta[istate] * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                                + soln_cell_avg[istate];
                        }
                    }
                }

                //project soln at quadrature points to dofs.
                for (int istate = 0; istate < nstate; istate++) {
                    soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_dofs[istate],
                        soln_basis_projection_oper.oneD_vol_operator);
                }

                //write limited solution dofs to the global solution vector.
                for (int istate = 0; istate < nstate; istate++) {
                    for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                        const unsigned int idof = istate * n_shape_fns + ishape;
                        this->solution[current_dofs_indices[idof]] = soln_dofs[istate][ishape];
                    }
                }
            }
        }

        /**********************************
        *
        * Maximum Principle Limiter Class
        *
        **********************************/
        // Constructor
        template <int dim, int nstate, typename real>
        MaximumPrincipleLimiter<dim, nstate, real>::MaximumPrincipleLimiter(
            const Parameters::AllParameters* const parameters_input)
            : all_parameters(parameters_input) {}

        template <int dim, int nstate, typename real>
        void MaximumPrincipleLimiter<dim, nstate, real>::get_global_max_and_min_of_solution(
            dealii::LinearAlgebra::distributed::Vector<double>      solution,
            dealii::DoFHandler<dim>                                 dof_handler,
            dealii::FESystem<dim, dim>& current_fe_ref,
            dealii::hp::QCollection<dim>                            volume_quadrature_collection)
        {
            for (auto soln_cell : this->dof_handler.active_cell_iterators()) {
                if (!soln_cell->is_locally_owned()) continue;

                std::vector<dealii::types::global_dof_index> current_dofs_indices;
                // Current reference element related to this physical cell
                const int i_fele = soln_cell->active_fe_index();
                const int poly_degree = i_fele;

                const dealii::FESystem<dim, dim>& current_fe_ref = this->fe_collection[poly_degree];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                soln_cell->get_dof_indices(current_dofs_indices);

                //Extract the local solution dofs in the cell from the global solution dofs
                if (this->global_max.size() < nstate && this->global_min.size() < nstate) {
                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        this->global_max.push_back(-1e9);
                        this->global_min.push_back(1e9);
                    }
                }

                //get solution coeff
                std::array<std::vector<real>, nstate> soln_dofs;
                const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
                for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                    const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
                    const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
                    if (ishape == 0) {
                        soln_dofs[istate].resize(n_shape_fns);
                    }
                    soln_dofs[istate][ishape] = this->solution[current_dofs_indices[idof]]; //
                    if (soln_dofs[istate][ishape] > this->global_max[istate])
                        this->global_max[istate] = soln_dofs[istate][ishape];
                    if (soln_dofs[istate][ishape] < this->global_min[istate])
                        this->global_min[istate] = soln_dofs[istate][ishape];
                }
            }

            for (unsigned int istate = 0; istate < nstate; ++istate) {
                std::cout << std::fixed;
                std::cout << std::setprecision(14);
                std::cout << "global_max:   " << this->global_max[istate] << "   global_min:   " << this->global_min[istate] << std::endl;
            }
        }

        template <int dim, int nstate, typename real>
        void MaximumPrincipleLimiter<dim, nstate, real>::limit(
            dealii::LinearAlgebra::distributed::Vector<double>      solution,
            dealii::DoFHandler<dim>                                 dof_handler,
            dealii::FESystem<dim, dim>& current_fe_ref,
            dealii::hp::QCollection<dim>                            volume_quadrature_collection)
        {
            //create 1D solution polynomial basis functions and corresponding projection operator
            //to interpolate the solution to the quadrature nodes, and to project it back to the
            //modal coefficients.
            const unsigned int init_grid_degree = this->high_order_grid->fe_system.tensor_degree();
            //Constructor for the operators
            OPERATOR::basis_functions<dim, 2 * dim> soln_basis(1, this->max_degree, init_grid_degree);
            OPERATOR::vol_projection_operator<dim, 2 * dim> soln_basis_projection_oper(1, this->max_degree, init_grid_degree);


            //build the oneD operator to perform interpolation/projection
            soln_basis.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);
            soln_basis_projection_oper.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);


            if (this->global_max.empty() && this->global_min.empty())
                this->get_global_max_and_min_of_solution();

            for (auto soln_cell : this->dof_handler.active_cell_iterators()) {
                if (!soln_cell->is_locally_owned()) continue;

                std::vector<dealii::types::global_dof_index> current_dofs_indices;
                // Current reference element related to this physical cell
                const int i_fele = soln_cell->active_fe_index();
                const int poly_degree = i_fele;

                const dealii::FESystem<dim, dim>& current_fe_ref = this->fe_collection[poly_degree];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                soln_cell->get_dof_indices(current_dofs_indices);

                //Extract the local solution dofs in the cell from the global solution dofs
                std::array<std::vector<real>, nstate> soln_dofs;
                const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
                std::array<real, nstate> local_max;
                std::array<real, nstate> local_min;
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    local_max[istate] = -1e9;
                    local_min[istate] = 1e9;

                    //allocate soln_dofs
                    soln_dofs[istate].resize(n_shape_fns);
                }

                for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                    const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
                    const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
                    soln_dofs[istate][ishape] = this->solution[current_dofs_indices[idof]]; //

                    if (soln_dofs[istate][ishape] > local_max[istate])
                        local_max[istate] = soln_dofs[istate][ishape];

                    if (soln_dofs[istate][ishape] < local_min[istate])
                        local_min[istate] = soln_dofs[istate][ishape];
                }

                const unsigned int n_quad_pts = this->volume_quadrature_collection[poly_degree].size();
                const std::vector<real>& quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
                //interpolate solution dofs to quadrature pts.
                //and apply integral for the soln avg
                std::array<real, nstate> soln_cell_avg;

                std::array<std::vector<real>, nstate> soln_at_q;

                for (int istate = 0; istate < nstate; istate++) {
                    soln_at_q[istate].resize(n_quad_pts);
                    soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q[istate],
                        soln_basis.oneD_vol_operator);
                }

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        soln_cell_avg[istate] += quad_weights[iquad]
                            * soln_at_q[istate][iquad];
                    }
                }

                //get theta value
                std::array<real, nstate> theta;
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

                //apply limiter on soln values at quadrature points
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        soln_at_q[istate][iquad] = theta[istate] * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                    }
                }

                //project soln at quadrature points to dofs.
                for (int istate = 0; istate < nstate; istate++) {
                    soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_dofs[istate],
                        soln_basis_projection_oper.oneD_vol_operator);
                }

                for (int istate = 0; istate < nstate; istate++) {
                    for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                        const unsigned int idof = istate * n_shape_fns + ishape;
                        this->solution[current_dofs_indices[idof]] = soln_dofs[istate][ishape];

                        if (this->solution[current_dofs_indices[idof]] > this->global_max[istate] + 1e-13) {
                            std::cout << " Solution exceeds global maximum   -   Aborting... Value:   " << this->solution[current_dofs_indices[idof]] << std::endl << std::flush;
                            this->pcout << "theta:   " << theta[istate] << "   local max:   " << local_max[istate] << "   soln_cell_avg:   " << soln_cell_avg[istate] << std::endl;
                            std::abort();
                        }
                        if (this->solution[current_dofs_indices[idof]] < this->global_min[istate] - 1e-13) {
                            std::cout << " Solution exceeds global minimum   -   Aborting... Value:   " << this->solution[current_dofs_indices[idof]] << std::endl << std::flush;
                            this->pcout << "theta:   " << theta[istate] << "   local_min:   " << local_min[istate] << "   soln_cell_avg:   " << soln_cell_avg[istate] << std::endl;
                            std::abort();
                        }
                    }
                }
            }
        }

        /**********************************
        *
        * Positivity Preserving Limiter Class
        *
        **********************************/
        // Constructor
        template <int dim, int nstate, typename real>
        PositivityPreservingLimiter<dim, nstate, real>::PositivityPreservingLimiter(
            const Parameters::AllParameters* const parameters_input)
            : all_parameters(parameters_input) {}

        template <int dim, int nstate, typename real>
        void PositivityPreservingLimiter<dim, nstate, real>::limit(
            dealii::LinearAlgebra::distributed::Vector<double>      solution,
            dealii::DoFHandler<dim>                                 dof_handler,
            dealii::FESystem<dim, dim>& current_fe_ref,
            dealii::hp::QCollection<dim>                            volume_quadrature_collection)
        {
            //create 1D solution polynomial basis functions and corresponding projection operator
            //to interpolate the solution to the quadrature nodes, and to project it back to the
            //modal coefficients.
            const unsigned int init_grid_degree = this->high_order_grid->fe_system.tensor_degree();

            //Constructor for the operators
            OPERATOR::basis_functions<dim, 2 * dim> soln_basis(1, this->max_degree, init_grid_degree);
            OPERATOR::vol_projection_operator<dim, 2 * dim> soln_basis_projection_oper(1, this->max_degree, init_grid_degree);

            //build the oneD operator to perform interpolation/projection
            soln_basis.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);
            soln_basis_projection_oper.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);

            for (auto soln_cell : this->dof_handler.active_cell_iterators()) {
                if (!soln_cell->is_locally_owned()) continue;

                std::vector<dealii::types::global_dof_index> current_dofs_indices;
                // Current reference element related to this physical cell
                const int i_fele = soln_cell->active_fe_index();
                const int poly_degree = i_fele;

                using FluxNodes = Parameters::AllParameters::FluxNodes;
                const FluxNodes flux_nodes_type = this->all_parameters->flux_nodes_type;

                const dealii::FESystem<dim, dim>& current_fe_ref = this->fe_collection[poly_degree];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                soln_cell->get_dof_indices(current_dofs_indices);

                //Extract the local solution dofs in the cell from the global solution dofs
                std::array<std::vector<real>, nstate> soln_dofs;

                const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
                std::array<real, nstate> local_min;
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    local_min[istate] = 1e9;

                    //allocate soln_dofs
                    soln_dofs[istate].resize(n_shape_fns);
                }

                for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                    const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
                    const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
                    soln_dofs[istate][ishape] = this->solution[current_dofs_indices[idof]]; //

                    if (soln_dofs[istate][ishape] < local_min[istate])
                        local_min[istate] = soln_dofs[istate][ishape];
                }

                const unsigned int n_quad_pts = this->volume_quadrature_collection[poly_degree].size();
                const std::vector<real>& quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
                //interpolate solution dofs to quadrature pts.
                //and apply integral for the soln avg
                std::array<real, nstate> soln_cell_avg;

                std::array<std::vector<real>, nstate> soln_at_q;

                for (int istate = 0; istate < nstate; istate++) {
                    soln_at_q[istate].resize(n_quad_pts);
                    soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q[istate],
                        soln_basis.oneD_vol_operator);
                }

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        if (flux_nodes_type == FluxNodes::GL) {
                            if (soln_at_q[istate][iquad] < local_min[istate])
                                local_min[istate] = soln_at_q[istate][iquad];
                        }
                        soln_cell_avg[istate] += quad_weights[iquad]
                            * soln_at_q[istate][iquad];
                    }
                }

                real eps = 1e-13;
                real gamma = 1.4;

                if (nstate == PHILIP_DIM + 2) {
                    // compute value of pressure at soln_cell_avg
                    real rho_avg = soln_cell_avg[0];
                    real tot_energy_avg = soln_cell_avg[nstate - 1];
                    real m_avg = soln_cell_avg[1];


                    real p_avg = (gamma - 1) * (tot_energy_avg - 0.5 * (((m_avg * m_avg) / rho_avg)));
                    if (dim > 1)
                        p_avg += -0.5 * ((soln_cell_avg[2] * soln_cell_avg[2]) / rho_avg);
                    if (dim > 2)
                        p_avg += -0.5 * ((soln_cell_avg[3] * soln_cell_avg[3]) / rho_avg);

                    //get epsilon (lower bound for density) for theta limiter
                    eps = std::min({ this->all_parameters->pos_eps, rho_avg, p_avg });
                    if (eps < 0) eps = this->all_parameters->pos_eps;
                }

                real theta = 1.0;
                if (soln_cell_avg[0] - local_min[0] > 1e-13)
                    theta = (soln_cell_avg[0] - local_min[0] == 0) ? 1.0 : std::min((soln_cell_avg[0] - eps) / (soln_cell_avg[0] - local_min[0]), 1.0);

                //apply limiter on density values at quadrature points
                for (unsigned int istate = 0; istate < 1/*nstate*/; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        if (isnan(soln_at_q[istate][iquad])) {
                            std::cout << " Density is NaN - Aborting... " << std::endl << std::flush;
                            std::abort();
                        }
                        soln_at_q[istate][iquad] = theta * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                    }
                }

                if (nstate == PHILIP_DIM + 2) {
                    std::vector< real > p_lim(n_quad_pts);
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        p_lim[iquad] = (gamma - 1) * (soln_at_q[nstate - 1][iquad] -
                            0.5 * ((soln_at_q[1][iquad] * soln_at_q[1][iquad]) / soln_at_q[0][iquad]));
                        if (dim > 1)
                            p_lim[iquad] += (gamma - 1) * (-0.5 * ((soln_at_q[2][iquad] * soln_at_q[2][iquad]) / soln_at_q[0][iquad]));
                        if (dim > 2)
                            p_lim[iquad] += (gamma - 1) * (-0.5 * ((soln_at_q[3][iquad] * soln_at_q[3][iquad]) / soln_at_q[0][iquad]));
                    }

                    std::vector<real> theta2(n_quad_pts, 1);
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        if (p_lim[iquad] >= eps)
                            theta2[iquad] = 1;
                        else {
                            real a = (soln_at_q[nstate - 1][iquad] - soln_cell_avg[nstate - 1]) * (soln_at_q[0][iquad] - soln_cell_avg[0]) - 0.5 * pow(soln_at_q[1][iquad] - soln_cell_avg[1], 2);

                            real b = soln_cell_avg[0] * (soln_at_q[nstate - 1][iquad] - soln_cell_avg[nstate - 1]) + soln_cell_avg[nstate - 1] * (soln_at_q[0][iquad] - soln_cell_avg[0])
                                - soln_cell_avg[1] * (soln_at_q[1][iquad] - soln_cell_avg[1]) - (eps / gamma) * (soln_at_q[0][iquad] - soln_cell_avg[0]);

                            real c = (soln_cell_avg[nstate - 1] * soln_cell_avg[0]) - 0.5 * pow(soln_cell_avg[1], 2) - (eps / gamma) * soln_cell_avg[0];

                            if (dim > 1) {
                                a -= 0.5 * pow(soln_at_q[2][iquad] - soln_cell_avg[2], 2);
                                b -= soln_cell_avg[2] * (soln_at_q[2][iquad] - soln_cell_avg[2]);
                                c -= 0.5 * pow(soln_cell_avg[2], 2);
                            }

                            if (dim > 2) {
                                a -= 0.5 * pow(soln_at_q[3][iquad] - soln_cell_avg[3], 2);
                                b -= soln_cell_avg[3] * (soln_at_q[3][iquad] - soln_cell_avg[3]);
                                c -= 0.5 * pow(soln_cell_avg[3], 2);
                            }

                            real discriminant = b * b - 4 * a * c;

                            if (discriminant >= 0) {
                                real t1 = (-b + sqrt(discriminant)) / (2 * a);
                                real t2 = (-b - sqrt(discriminant)) / (2 * a);
                                theta2[iquad] = std::min(t1, t2);
                            }
                            else {
                                theta2[iquad] = 0; // (-b) / (2 * a);
                            }
                        }
                    }

                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                            soln_at_q[istate][iquad] = theta2[iquad] * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                                + soln_cell_avg[istate];
                        }
                    }
                }

                //project soln at quadrature points to dofs.
                for (int istate = 0; istate < nstate; istate++) {
                    soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_dofs[istate],
                        soln_basis_projection_oper.oneD_vol_operator);
                }

                // //write limited solution dofs to the global solution vector.
                for (int istate = 0; istate < nstate; istate++) {
                    for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                        const unsigned int idof = istate * n_shape_fns + ishape;
                        this->solution[current_dofs_indices[idof]] = soln_dofs[istate][ishape]; //

                        if (istate == 0) {
                            if (this->solution[current_dofs_indices[idof]] < 0) {
                                std::cout << "Density is a negative value - Aborting... " << std::endl << this->solution[current_dofs_indices[idof]] << std::endl << std::flush;
                                std::abort();
                            }
                        }
                    }
                }
            }
        }


        /**********************************
        *
        * Positivity Preserving Limiter Robust Class
        *
        **********************************/
        // Constructor
        template <int dim, int nstate, typename real>
        PositivityPreservingLimiterRobust<dim, nstate, real>::PositivityPreservingLimiterRobust(
            const Parameters::AllParameters* const parameters_input)
            : all_parameters(parameters_input) {}

        template <int dim, int nstate, typename real>
        void PositivityPreservingLimiterRobust<dim, nstate, real>::limit(
            dealii::LinearAlgebra::distributed::Vector<double>      solution,
            dealii::DoFHandler<dim>                                 dof_handler,
            dealii::FESystem<dim, dim>& current_fe_ref,
            dealii::hp::QCollection<dim>                            volume_quadrature_collection)
        {
            //create 1D solution polynomial basis functions and corresponding projection operator
            //to interpolate the solution to the quadrature nodes, and to project it back to the
            //modal coefficients.
            const unsigned int init_grid_degree = this->high_order_grid->fe_system.tensor_degree();

            //Constructor for the operators
            OPERATOR::basis_functions<dim, 2 * dim> soln_basis(1, this->max_degree, init_grid_degree);
            OPERATOR::vol_projection_operator<dim, 2 * dim> soln_basis_projection_oper(1, this->max_degree, init_grid_degree);

            //build the oneD operator to perform interpolation/projection
            soln_basis.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);
            soln_basis_projection_oper.build_1D_volume_operator(this->oneD_fe_collection_1state[this->max_degree], this->oneD_quadrature_collection[this->max_degree]);

            for (auto soln_cell : this->dof_handler.active_cell_iterators()) {
                if (!soln_cell->is_locally_owned()) continue;

                std::vector<dealii::types::global_dof_index> current_dofs_indices;
                // Current reference element related to this physical cell
                const int i_fele = soln_cell->active_fe_index();
                const int poly_degree = i_fele;

                const dealii::FESystem<dim, dim>& current_fe_ref = this->fe_collection[poly_degree];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                soln_cell->get_dof_indices(current_dofs_indices);

                //Extract the local solution dofs in the cell from the global solution dofs
                std::array<std::vector<real>, nstate> soln_dofs;

                const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
                std::array<real, nstate> local_min;
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    local_min[istate] = 1e9;

                    //allocate soln_dofs
                    soln_dofs[istate].resize(n_shape_fns);
                }

                for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                    const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
                    const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
                    soln_dofs[istate][ishape] = this->solution[current_dofs_indices[idof]];

                    if (soln_dofs[istate][ishape] < local_min[istate])
                        local_min[istate] = soln_dofs[istate][ishape];
                }

                const unsigned int n_quad_pts = this->volume_quadrature_collection[poly_degree].size();
                const std::vector<real>& quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
                //interpolate solution dofs to quadrature pts.
                //and apply integral for the soln avg
                std::array<real, nstate> soln_cell_avg;

                std::array<std::vector<real>, nstate> soln_at_q;

                for (int istate = 0; istate < nstate; istate++) {
                    soln_at_q[istate].resize(n_quad_pts);
                    soln_basis.matrix_vector_mult_1D(soln_dofs[istate], soln_at_q[istate],
                        soln_basis.oneD_vol_operator);
                }

                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        if (soln_at_q[istate][iquad] < local_min[istate])
                            local_min[istate] = soln_at_q[istate][iquad];

                        soln_cell_avg[istate] += quad_weights[iquad]
                            * soln_at_q[istate][iquad];
                    }
                }

                real eps = 1e-13;
                real gamma = 1.4;

                if (nstate == PHILIP_DIM + 2) {
                    // compute value of pressure at soln_cell_avg
                    real rho_avg = soln_cell_avg[0];
                    real tot_energy_avg = soln_cell_avg[nstate - 1];
                    real m_avg = soln_cell_avg[1];


                    real p_avg = (gamma - 1) * (tot_energy_avg - 0.5 * (((m_avg * m_avg) / rho_avg)));
                    if (dim > 1)
                        p_avg += -0.5 * (gamma - 1) * ((soln_cell_avg[2] * soln_cell_avg[2]) / rho_avg);
                    if (dim > 2)
                        p_avg += -0.5 * (gamma - 1) * ((soln_cell_avg[3] * soln_cell_avg[3]) / rho_avg);

                    //get epsilon (lower bound for rho) for theta limiter
                    eps = std::min({ this->all_parameters->pos_eps, rho_avg, p_avg });
                    if (eps < 0) eps = this->all_parameters->pos_eps;
                }

                real theta = 1.0;
                if (soln_cell_avg[0] - local_min[0] > 1e-13)
                    theta = (soln_cell_avg[0] - local_min[0] == 0) ? 1.0 : std::min((soln_cell_avg[0] - eps) / (soln_cell_avg[0] - local_min[0]), 1.0);

                if (theta < 0 || theta > 1)
                    theta = 0;

                //apply limiter on density values at quadrature points
                for (unsigned int istate = 0; istate < 1/*nstate*/; ++istate) {
                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        if (isnan(soln_at_q[istate][iquad])) {
                            std::cout << " Density is NaN - Aborting... " << std::endl << std::flush;
                            std::abort();
                        }

                        soln_at_q[istate][iquad] = theta * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                    }
                }

                if (nstate == PHILIP_DIM + 2) {
                    real p_lim = 0.0;
                    real p_avg = 0.0;
                    std::vector<real> t2(n_quad_pts, 1);
                    real theta2 = 1.0;

                    p_avg = (gamma - 1) * (soln_cell_avg[nstate - 1] - 0.5 * (pow(soln_cell_avg[1], 2) / soln_cell_avg[0]));
                    if (dim > 1)
                        p_avg += -0.5 * (gamma - 1) * (pow(soln_cell_avg[2], 2) / soln_cell_avg[0]);
                    if (dim > 2)
                        p_avg += -0.5 * (gamma - 1) * (pow(soln_cell_avg[3], 2) / soln_cell_avg[0]);

                    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                        p_lim = 0.0;
                        p_lim = (gamma - 1) * (soln_at_q[nstate - 1][iquad] -
                            0.5 * (pow(soln_at_q[1][iquad], 2) / soln_at_q[0][iquad]));
                        if (dim > 1)
                            p_lim += (gamma - 1) * (-0.5 * (pow(soln_at_q[2][iquad], 2) / soln_at_q[0][iquad]));
                        if (dim > 2)
                            p_lim += (gamma - 1) * (-0.5 * (pow(soln_at_q[3][iquad], 2) / soln_at_q[0][iquad]));

                        if (p_lim >= 0)
                            t2[iquad] = 1;
                        else
                            t2[iquad] = p_avg / (p_avg - p_lim);

                        if (t2[iquad] != 1) {
                            if (t2[iquad] >= 0 && t2[iquad] <= 1 && t2[iquad] < theta2) {
                                theta2 = t2[iquad];
                            }
                        }
                    }


                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                            soln_at_q[istate][iquad] = theta2 * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                                + soln_cell_avg[istate];
                        }
                    }
                }

                //project soln at quadrature points to dofs.
                for (int istate = 0; istate < nstate; istate++) {
                    soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_dofs[istate],
                        soln_basis_projection_oper.oneD_vol_operator);
                }

                // //write limited solution dofs to the global solution vector.
                for (int istate = 0; istate < nstate; istate++) {
                    for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                        const unsigned int idof = istate * n_shape_fns + ishape;
                        this->solution[current_dofs_indices[idof]] = soln_dofs[istate][ishape]; //

                        if (istate == 0) {
                            if (this->solution[current_dofs_indices[idof]] < 0) {
                                std::cout << "Density is a negative value - Aborting... " << std::endl << this->solution[current_dofs_indices[idof]] << std::endl << std::flush;
                                std::abort();
                            }
                        }
                    }
                }
            }
        }

        template class BoundPreservingLimiter <PHILIP_DIM, 1, double>;
        template class BoundPreservingLimiter <PHILIP_DIM, 2, double>;
        template class BoundPreservingLimiter <PHILIP_DIM, 3, double>;
        template class BoundPreservingLimiter <PHILIP_DIM, 4, double>;
        template class BoundPreservingLimiter <PHILIP_DIM, 5, double>;
        template class BoundPreservingLimiter <PHILIP_DIM, 6, double>;

        template class TVBLimiter <PHILIP_DIM, 1, double>;
        template class TVBLimiter <PHILIP_DIM, 2, double>;
        template class TVBLimiter <PHILIP_DIM, 3, double>;
        template class TVBLimiter <PHILIP_DIM, 4, double>;
        template class TVBLimiter <PHILIP_DIM, 5, double>;
        template class TVBLimiter <PHILIP_DIM, 6, double>;

        template class MaximumPrincipleLimiter <PHILIP_DIM, 1, double>;
        template class MaximumPrincipleLimiter <PHILIP_DIM, 2, double>;
        template class MaximumPrincipleLimiter <PHILIP_DIM, 3, double>;
        template class MaximumPrincipleLimiter <PHILIP_DIM, 4, double>;
        template class MaximumPrincipleLimiter <PHILIP_DIM, 5, double>;
        template class MaximumPrincipleLimiter <PHILIP_DIM, 6, double>;

        template class PositivityPreservingLimiter <PHILIP_DIM, 1, double>;
        template class PositivityPreservingLimiter <PHILIP_DIM, 2, double>;
        template class PositivityPreservingLimiter <PHILIP_DIM, 3, double>;
        template class PositivityPreservingLimiter <PHILIP_DIM, 4, double>;
        template class PositivityPreservingLimiter <PHILIP_DIM, 5, double>;
        template class PositivityPreservingLimiter <PHILIP_DIM, 6, double>;

        template class PositivityPreservingLimiterRobust <PHILIP_DIM, 1, double>;
        template class PositivityPreservingLimiterRobust <PHILIP_DIM, 2, double>;
        template class PositivityPreservingLimiterRobust <PHILIP_DIM, 3, double>;
        template class PositivityPreservingLimiterRobust <PHILIP_DIM, 4, double>;
        template class PositivityPreservingLimiterRobust <PHILIP_DIM, 5, double>;
        template class PositivityPreservingLimiterRobust <PHILIP_DIM, 6, double>;
    } // LIMITER namespace
} // PHiLiP namespace