#include "positivity_preserving_limiter.h"
#include "tvb_limiter.h"
#include <eigen/unsupported/Eigen/Polynomials>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
/**********************************
*
* Positivity Preserving Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
PositivityPreservingLimiter<dim, nstate, real>::PositivityPreservingLimiter(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiterState<dim,nstate,real>::BoundPreservingLimiterState(parameters_input)
    , flow_solver_param(parameters_input->flow_solver_param)
    , dx((flow_solver_param.grid_xmax-flow_solver_param.grid_xmin)/flow_solver_param.number_of_grid_elements_x)
    , dy((flow_solver_param.grid_ymax-flow_solver_param.grid_ymin)/flow_solver_param.number_of_grid_elements_y)
    , dz((flow_solver_param.grid_zmax-flow_solver_param.grid_zmin)/flow_solver_param.number_of_grid_elements_z)
{
    // Create pointer to Euler Physics to compute pressure if pde_type==euler
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PDE_enum pde_type = parameters_input->pde_type;

    std::shared_ptr< ManufacturedSolutionFunction<dim, real> >  manufactured_solution_function
        = ManufacturedSolutionFactory<dim, real>::create_ManufacturedSolution(parameters_input, nstate);

    if (pde_type == PDE_enum::euler && nstate == dim + 2) {
        euler_physics = std::make_shared < Physics::Euler<dim, nstate, real> >(
            parameters_input,
            parameters_input->euler_param.ref_length,
            parameters_input->euler_param.gamma_gas,
            parameters_input->euler_param.mach_inf,
            parameters_input->euler_param.angle_of_attack,
            parameters_input->euler_param.side_slip_angle,
            manufactured_solution_function,
            parameters_input->two_point_num_flux_type);
    }
    else {
        std::cout << "Error: Positivity-Preserving Limiter can only be applied for pde_type==euler" << std::endl;
        std::abort();
    }

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

    if(dim >= 2 && (flow_solver_param.number_of_grid_elements_x == 1 || flow_solver_param.number_of_grid_elements_y == 1)) {
        std::cout << "Error: number_of_grid_elements must be passed for all directions to use PPL Limiter." << std::endl;
        std::abort();
    }

    if(dim == 3 && flow_solver_param.number_of_grid_elements_z == 1) {
        std::cout << "Error: number_of_grid_elements must be passed for all directions to use PPL Limiter." << std::endl;
        std::abort();
    }
}

template <int dim, int nstate, typename real>
std::vector<real> PositivityPreservingLimiter<dim, nstate, real>::get_theta2_Zhang2010(
    const std::vector< real >&                      p_lim,
    const std::array<real, nstate>&                 soln_cell_avg,
    const std::array<std::vector<real>, nstate>&    soln_at_q,
    const unsigned int                              n_quad_pts,
    const double                                    eps,
    const double                                    gamma)
{
    std::vector<real> theta2(n_quad_pts, 1); // Value used to linearly scale state variables
    Eigen::PolynomialSolver<double, 2> solver; // Solver to find smallest root

    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        if (p_lim[iquad] >= eps) {
            theta2[iquad] = 1;
        }
        else {
            real s_coeff1 = (soln_at_q[nstate - 1][iquad] - soln_cell_avg[nstate - 1]) * (soln_at_q[0][iquad] - soln_cell_avg[0]) - 0.5 * pow(soln_at_q[1][iquad] - soln_cell_avg[1], 2);

            real s_coeff2 = soln_cell_avg[0] * (soln_at_q[nstate - 1][iquad] - soln_cell_avg[nstate - 1]) + soln_cell_avg[nstate - 1] * (soln_at_q[0][iquad] - soln_cell_avg[0])
                - soln_cell_avg[1] * (soln_at_q[1][iquad] - soln_cell_avg[1]) - (eps / gamma) * (soln_at_q[0][iquad] - soln_cell_avg[0]);

            real s_coeff3 = (soln_cell_avg[nstate - 1] * soln_cell_avg[0]) - 0.5 * pow(soln_cell_avg[1], 2) - (eps / gamma) * soln_cell_avg[0];

            if (dim > 1) {
                s_coeff1 -= 0.5 * pow(soln_at_q[2][iquad] - soln_cell_avg[2], 2);
                s_coeff2 -= soln_cell_avg[2] * (soln_at_q[2][iquad] - soln_cell_avg[2]);
                s_coeff3 -= 0.5 * pow(soln_cell_avg[2], 2);
            }

            if (dim > 2) {
                s_coeff1 -= 0.5 * pow(soln_at_q[3][iquad] - soln_cell_avg[3], 2);
                s_coeff2 -= soln_cell_avg[3] * (soln_at_q[3][iquad] - soln_cell_avg[3]);
                s_coeff3 -= 0.5 * pow(soln_cell_avg[3], 2);
            }

            Eigen::Vector3d coeff(s_coeff3, s_coeff2, s_coeff1);
            solver.compute(coeff);
            const Eigen::PolynomialSolver<double, 2>::RootType &r = solver.smallestRoot();
            theta2[iquad] = r.real();
        }
    }

    return theta2;
}

template <int dim, int nstate, typename real>
real PositivityPreservingLimiter<dim, nstate, real>::get_theta2_Wang2012(
    const std::array<std::vector<real>, nstate>&    soln_at_q,
    const unsigned int                              n_quad_pts,
    const double                                    p_avg)
{
    std::vector<real> t2(n_quad_pts, 1);
    real theta2 = 1.0; // Value used to linearly scale state variables 
    std::array<real, nstate> soln_at_iquad;

    // Obtain theta2 value
    for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_at_iquad[istate] = soln_at_q[istate][iquad];
        }
        real p_lim = 0;

        if (nstate == dim + 2)
            p_lim = euler_physics->compute_pressure(soln_at_iquad);

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

    return theta2;
}

template <int dim, int nstate, typename real>
real PositivityPreservingLimiter<dim, nstate, real>::get_density_scaling_value(
    const double    density_avg,
    const double    density_min,
    const double    lower_bound,
    const double    p_avg)
{
    // Get epsilon (lower bound for rho) for theta limiter
    real eps = std::min({ lower_bound, density_avg, p_avg });
    if (eps < 0) eps = lower_bound;

    real theta = 1.0; // Value used to linearly scale density 
    if (density_avg - density_min > 1e-13)
        theta = (density_avg - density_min == 0) ? 1.0 : std::min((density_avg - eps) / (density_avg - density_min), 1.0);

    if (theta < 0 || theta > 1)
        theta = 0;

    return theta;
}

template <int dim, int nstate, typename real>
void PositivityPreservingLimiter<dim, nstate, real>::write_limited_solution(
    dealii::LinearAlgebra::distributed::Vector<double>& solution,
    const std::array<std::vector<real>, nstate>& soln_coeff,
    const unsigned int                                      n_shape_fns,
    const std::vector<dealii::types::global_dof_index>& current_dofs_indices)
{
    // Write limited solution dofs to the global solution vector.
    for (int istate = 0; istate < nstate; istate++) {
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            const unsigned int idof = istate * n_shape_fns + ishape;
            solution[current_dofs_indices[idof]] = soln_coeff[istate][ishape]; //

            // Verify that positivity of density is preserved after application of theta2 limiter
            if (istate == 0 && solution[current_dofs_indices[idof]] < 0) {
                std::cout << "Error: Density is a negative value - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }

            // Verify that positivity of Total Energy is preserved after application of theta2 limiter
            if (istate == (nstate - 1) && solution[current_dofs_indices[idof]] < 0) {
                std::cout << "Error: Total Energy is a negative value - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }

            // Verify that the solution values haven't been changed to NaN as a result of all quad pts in a cell having negative density 
            // (all quad pts having negative density would result in the local maximum convective eigenvalue being zero leading to division by zero)
            if (isnan(solution[current_dofs_indices[idof]])) {
                std::cout << "Error: Solution is NaN - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                std::abort();
            }
        }
    }
}

template <int dim, int nstate, typename real>
std::array<real, nstate> PositivityPreservingLimiter<dim, nstate, real>::get_soln_cell_avg_PPL(
    const std::array<std::array<std::vector<real>, nstate>, dim>&        soln_at_q,
    const unsigned int                                                   n_quad_pts,
    const std::vector<real>&                                             quad_weights_GLL,
    const std::vector<real>&                                             quad_weights_GL,
    double&                                                              dt)
{
    std::array<real, nstate> soln_cell_avg;

    // Obtain solution cell average
    if (dim == 1) {
        soln_cell_avg = get_soln_cell_avg(soln_at_q[0], n_quad_pts, quad_weights_GLL);
    } else if (dim > 1) {
        std::array<std::array<real, nstate>,dim> soln_cell_avg_dim;

        for(unsigned int idim = 0; idim < dim; ++idim) {
            for(unsigned int istate = 0; istate < nstate; ++istate) {
                soln_cell_avg_dim[idim][istate] = 0;
            }
        }

        if constexpr (dim == 2) {
            // Calculating average in x-dir - GLL used for x direction to include surface nodes, GL for rest
            for(unsigned int istate = 0; istate < nstate; ++istate) {
                unsigned int quad_pt = 0;
                for(unsigned int iquad=0; iquad<quad_weights_GLL.size(); ++iquad) {
                    for(unsigned int jquad=0; jquad<quad_weights_GL.size(); ++jquad) {
                        soln_cell_avg_dim[0][istate] += quad_weights_GLL[iquad]*quad_weights_GL[jquad]*soln_at_q[0][istate][quad_pt];
                            quad_pt++;
                    }
                }
            }

            // Calculating average in y-dir - GLL used for y direction to include surface nodes, GL for rest
            for(unsigned int istate = 0; istate < nstate; ++istate) {
                unsigned int quad_pt = 0;
                for(unsigned int iquad=0; iquad<quad_weights_GL.size(); ++iquad) {
                    for(unsigned int jquad=0; jquad<quad_weights_GLL.size(); ++jquad) {
                        soln_cell_avg_dim[1][istate] += quad_weights_GL[iquad]*quad_weights_GLL[jquad]*soln_at_q[1][istate][quad_pt];
                            quad_pt++;
                    }
                }
            }
        }

        if constexpr (dim == 3) {
            // Calculating average in x-dir - GLL used for x direction to include surface nodes, GL for rest
            for(unsigned int istate = 0; istate < nstate; ++istate) {
                unsigned int quad_pt = 0;
                for(unsigned int iquad=0; iquad<quad_weights_GLL.size(); ++iquad) {
                    for(unsigned int jquad=0; jquad<quad_weights_GL.size(); ++jquad) {
                        for(unsigned int kquad=0; kquad<quad_weights_GL.size(); ++kquad)
                            soln_cell_avg_dim[0][istate] += quad_weights_GLL[iquad]*quad_weights_GL[jquad]*quad_weights_GL[kquad]*soln_at_q[0][istate][quad_pt];
                                quad_pt++;
                    }
                }
            }

            // Calculating average in y-dir - GLL used for y direction to include surface nodes, GL for rest
            for(unsigned int istate = 0; istate < nstate; ++istate) {
                unsigned int quad_pt = 0;
                for(unsigned int iquad=0; iquad<quad_weights_GL.size(); ++iquad) {
                    for(unsigned int jquad=0; jquad<quad_weights_GLL.size(); ++jquad) {
                        for(unsigned int kquad=0; kquad<quad_weights_GL.size(); ++kquad)
                            soln_cell_avg_dim[1][istate] += quad_weights_GL[iquad]*quad_weights_GLL[jquad]*quad_weights_GL[kquad]*soln_at_q[1][istate][quad_pt];
                                quad_pt++;
                    }
                }
            }

            // Calculating average in z-dir - GLL used for z direction to include surface nodes, GL for rest
            for(unsigned int istate = 0; istate < nstate; ++istate) {
                unsigned int quad_pt = 0;
                for(unsigned int iquad=0; iquad<quad_weights_GL.size(); ++iquad) {
                    for(unsigned int jquad=0; jquad<quad_weights_GL.size(); ++jquad) {
                        for(unsigned int kquad=0; kquad<quad_weights_GLL.size(); ++kquad)
                            soln_cell_avg_dim[2][istate] += quad_weights_GL[iquad]*quad_weights_GL[jquad]*quad_weights_GLL[kquad]*soln_at_q[2][istate][quad_pt];
                                quad_pt++;
                    }
                }
            }
        }

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_cell_avg[istate] = 0;
        }

        // Values required to weight the averages of each set of mixed nodes (refer to Eqn3.8 in Zhang,Shu paper)
        const real lambda_1 = dt/this->dx; const real lambda_2 = dt/this->dy; real lambda_3 = 0.0;
        if constexpr(dim == 3)
            lambda_3 = dt/this->dz;

        real max_local_wave_speed_1 = 0.0;
        real max_local_wave_speed_2 = 0.0;
        real max_local_wave_speed_3 = 0.0;
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            std::array<real,nstate> local_soln_at_q_1;
            std::array<real,nstate> local_soln_at_q_2;
            std::array<real,nstate> local_soln_at_q_3;
            for(unsigned int istate = 0; istate < nstate; ++istate){
                local_soln_at_q_1[istate] = soln_at_q[0][istate][iquad];
                local_soln_at_q_2[istate] = soln_at_q[1][istate][iquad];
                if(dim == 3)
                    local_soln_at_q_3[istate] = soln_at_q[2][istate][iquad];
                else
                    local_soln_at_q_3[istate] = 0.0;
            }
            // Update the maximum local wave speed (i.e. convective eigenvalue)
            const real local_wave_speed_1 = this->euler_physics->max_convective_eigenvalue(local_soln_at_q_1);
            const real local_wave_speed_2 = this->euler_physics->max_convective_eigenvalue(local_soln_at_q_2);

            real local_wave_speed_3 = 0.0;
            if(dim == 3)
                local_wave_speed_3 = this->euler_physics->max_convective_eigenvalue(local_soln_at_q_3);

            if(local_wave_speed_1 > max_local_wave_speed_1) max_local_wave_speed_1 = local_wave_speed_1;
            if(local_wave_speed_2 > max_local_wave_speed_2) max_local_wave_speed_2 = local_wave_speed_2;
            if(dim == 3 && local_wave_speed_3 > max_local_wave_speed_3) max_local_wave_speed_3 = local_wave_speed_3;

        }

        real mu = max_local_wave_speed_1*lambda_1 + max_local_wave_speed_2*lambda_2 + max_local_wave_speed_3*lambda_3;
        real avg_weight_1 = (max_local_wave_speed_1*lambda_1)/mu;
        real avg_weight_2 = (max_local_wave_speed_2*lambda_2)/mu;
        real avg_weight_3 = (max_local_wave_speed_3*lambda_3)/mu;

        for (unsigned int istate = 0; istate < nstate; istate++) {
            soln_cell_avg[istate] = avg_weight_1*soln_cell_avg_dim[0][istate] + avg_weight_2*soln_cell_avg_dim[1][istate];
            if(dim == 3)
                soln_cell_avg[istate] += avg_weight_3*soln_cell_avg_dim[2][istate];

            if (isnan(soln_cell_avg[istate])) {
                std::cout << "Error: Solution Cell Avg is NaN - Aborting... " << std::endl << std::flush;
                std::abort();
            }
        }
    }
    return soln_cell_avg;
}

template <int dim, int nstate, typename real>
void PositivityPreservingLimiter<dim, nstate, real>::limit(
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

    //create 1D solution polynomial basis functions to interpolate the solution to the quadrature nodes
    const unsigned int init_grid_degree = grid_degree;

    // Construct 1D Quad Points
    dealii::QGauss<1> oneD_quad_GL(max_degree + 1);
    dealii::QGaussLobatto<1> oneD_quad_GLL(max_degree + 1);

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_GLL(1, max_degree, init_grid_degree);
    soln_basis_GLL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quad_GLL);
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis_GL(1, max_degree, init_grid_degree);
    soln_basis_GL.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quad_GL);

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
        real local_min_density = 1e6;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        bool nan_check = false;
        // Allocate solution dofs and set local min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]];

            if (isnan(soln_coeff[istate][ishape])) {
                nan_check = true;
            }
        }

        const unsigned int n_quad_pts = n_shape_fns;

        if (nan_check) {
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                std::cout << "Error: Density passed to limiter is NaN - Aborting... " << std::endl;

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    std::cout << soln_coeff[istate][iquad] << "    ";
                }

                std::cout << std::endl;

                std::abort();
            }  
        }

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
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                if (soln_coeff[0][iquad] < local_min_density)
                    local_min_density = soln_coeff[0][iquad];
                if (soln_at_q[idim][0][iquad] < local_min_density)
                    local_min_density = soln_at_q[idim][0][iquad];
            }
        }

        std::vector< real > GLL_weights = oneD_quad_GLL.get_weights();
        std::vector< real > GL_weights = oneD_quad_GL.get_weights();
        std::array<real, nstate> soln_cell_avg;
        // Obtain solution cell average
        soln_cell_avg = get_soln_cell_avg_PPL(soln_at_q, n_quad_pts, oneD_quad_GLL.get_weights(), oneD_quad_GL.get_weights(), dt);

        real lower_bound = this->all_parameters->limiter_param.min_density;
        real p_avg = 1e-13;

        if (nstate == dim + 2) {
            // Compute average value of pressure using soln_cell_avg
            p_avg = euler_physics->compute_pressure(soln_cell_avg);
        }
        
        // Obtain value used to linearly scale density
        real theta = get_density_scaling_value(soln_cell_avg[0], local_min_density, lower_bound, p_avg);

        // Apply limiter on density values at quadrature points
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            soln_coeff[0][ishape] = theta*(soln_coeff[0][ishape] - soln_cell_avg[0]) + soln_cell_avg[0];
        }

        // Interpolate new density values to mixed quadrature points
        if(dim >= 1) {
            soln_basis_GLL.matrix_vector_mult(soln_coeff[0], soln_at_q[0][0],
                soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
        }

        if(dim >= 2) {
            soln_basis_GLL.matrix_vector_mult(soln_coeff[0], soln_at_q[1][0],
                soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator);
        }

        if(dim == 3) {
            soln_basis_GLL.matrix_vector_mult(soln_coeff[0], soln_at_q[2][0],
                soln_basis_GL.oneD_vol_operator, soln_basis_GL.oneD_vol_operator, soln_basis_GLL.oneD_vol_operator);
        }


        real theta2 = 1.0;
        using limiter_enum = Parameters::LimiterParam::LimiterType;
        limiter_enum limiter_type = this->all_parameters->limiter_param.bound_preserving_limiter;

        if (limiter_type == limiter_enum::positivity_preservingWang2012 && nstate == dim + 2) {
            std::array<real, dim> theta2_quad;
            for(unsigned int idim = 0; idim < dim; ++idim) {
                theta2_quad[idim] = get_theta2_Wang2012(soln_at_q[idim], n_quad_pts, p_avg);
            }

            for(unsigned int idim = 0; idim < dim; ++idim) {
                if(theta2_quad[idim] < theta2)
                    theta2 = theta2_quad[idim];
            }

            real theta2_soln = get_theta2_Wang2012(soln_coeff, n_quad_pts, p_avg);
            if(theta2_soln < theta2)
                    theta2 = theta2_soln;

            // Limit values at quadrature points
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    soln_coeff[istate][iquad] = theta2 * (soln_coeff[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                }
            }
        }

        if (limiter_type == limiter_enum::positivity_preservingZhang2010 && nstate == dim + 2) {

            std::array<std::vector< real >, dim> p_lim_quad;
            std::array<real, nstate> soln_at_iquad;

            for(unsigned int idim = 0; idim < dim; ++idim) {
                p_lim_quad[idim].resize(n_quad_pts);
                // Compute pressure at quadrature points
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    for (unsigned int istate = 0; istate < nstate; ++istate) {
                        soln_at_iquad[istate] = soln_at_q[idim][istate][iquad];
                    }
                    p_lim_quad[idim][iquad] = euler_physics->compute_pressure(soln_at_iquad);
                }
            }

            std::array<std::vector< real >, dim> theta2_quad;
            // Obtain value used to linearly scale state variables
            for(unsigned int idim = 0; idim < dim; ++idim) {
                theta2_quad[idim].resize(n_quad_pts);
                theta2_quad[idim] = get_theta2_Zhang2010(p_lim_quad[idim], soln_cell_avg, soln_at_q[idim], n_quad_pts, lower_bound, euler_physics->gam);
            }

            // Compute pressure at solution points
            std::vector< real > p_lim;
            p_lim.resize(n_quad_pts);
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                for (unsigned int istate = 0; istate < nstate; ++istate) {
                    soln_at_iquad[istate] = soln_coeff[istate][iquad];
                }
                p_lim[iquad] = euler_physics->compute_pressure(soln_at_iquad);
            }
            std::vector<real> theta2_soln = get_theta2_Zhang2010(p_lim, soln_cell_avg, soln_coeff, n_quad_pts, lower_bound, euler_physics->gam);

            // Limit values at quadrature points
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    real min_theta2_quad = 1e6;
                    for(unsigned int idim = 0; idim < dim; ++idim) {
                        if(theta2_quad[idim][iquad] < min_theta2_quad)
                            min_theta2_quad = theta2_quad[idim][iquad];
                    }

                    theta2 = std::min({ min_theta2_quad, theta2_soln[iquad] });
                    soln_coeff[istate][iquad] = theta2 * (soln_coeff[istate][iquad] - soln_cell_avg[istate])
                            + soln_cell_avg[istate];
                }
            }
        }

        if (isnan(theta2)) {
            std::cout << "Error: Theta2 is NaN - Aborting... " << std::endl << theta2 << std::endl << std::flush;
            std::abort();
        }

        // Write limited solution back and verify that positivity of density is satisfied
        write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);
    }
}

template class PositivityPreservingLimiter <PHILIP_DIM, PHILIP_DIM + 2, double>;
} // PHiLiP namespace