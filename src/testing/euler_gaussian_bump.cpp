#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "euler_gaussian_bump.h"
#include "mesh/grids/gaussian_bump.h"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerGaussianBump<dim,nstate>::EulerGaussianBump(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerGaussianBump<dim,nstate>
::run_test () const
{
    int convergence_order_achieved = 0;
    double run_test_output = run_euler_gaussian_bump(); // run_euler_gaussian_bump() can return either enthalpy or the convergence order
    if (abs(run_test_output) > 1e-15) // If run_euler_gaussian_bump() returns non zero
    {
        convergence_order_achieved = 1;  // test failed
    }
    return convergence_order_achieved;
}

template<int dim, int nstate>
double EulerGaussianBump<dim,nstate>
::run_euler_gaussian_bump() const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(dim == 2, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;

    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    std::string error_string;
    bool has_residual_converged = true;
    double artificial_dissipation_max_coeff = 0.0;
    double last_error=10;
    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input;

        std::vector<double> error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        std::vector<unsigned int> n_subdivisions(dim);
        n_subdivisions[1] = n_1d_cells[0]; // y-direction
        n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

        // const double channel_length = 3.0;
        // const double channel_height = 0.8;
        // Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);

        // const double solution_degree = poly_degree;
        // const double grid_degree = 3;
        // // Create DG object
        // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, &grid);

        // // Initialize coarse grid solution with free-stream
        // dg->allocate_system ();
        // dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // // Create ODE solver and ramp up the solution from p0
        // std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        // ode_solver->initialize_steady_polynomial_ramping (poly_degree);

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            //if (igrid!=0) {
            //    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
            //    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim>> solution_transfer(dg->dof_handler);
            //    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
            //    dg->high_order_grid->prepare_for_coarsening_and_refinement();
            //    grid->refine_global (1);
            //    dg->high_order_grid->execute_coarsening_and_refinement(true);
            //    dg->allocate_system ();
            //    dg->solution.zero_out_ghosts();
            //    solution_transfer.interpolate(dg->solution);
            //    dg->solution.update_ghost_values();
            //}

            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));

            const double channel_length = 3.0;
            const double channel_height = 0.8;
            Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);
            grid->refine_global(igrid);

            const double solution_degree = poly_degree;
            const double grid_degree = solution_degree+1;
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, grid);

            // Initialize coarse grid solution with free-stream
            dg->allocate_system ();
            dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Create ODE solver and ramp up the solution from p0
            std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
            ode_solver->initialize_steady_polynomial_ramping (poly_degree);

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            //dealii::MappingQ<dim> mapping(dg->max_degree+overintegrate);
            //const dealii::MappingManifold<dim,dim> mapping;
            const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
            dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0;

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

            // Integrate solution error and output error
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;
                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    double unumerical, uexact;
                    if(param.artificial_dissipation_param.use_enthalpy_error)
                    {
                        error_string = "L2_enthalpy_error";
                        const double pressure = euler_physics_double.compute_pressure(soln_at_q);
                        unumerical = euler_physics_double.compute_specific_enthalpy(soln_at_q,pressure);
                        uexact = euler_physics_double.gam*euler_physics_double.pressure_inf/euler_physics_double.density_inf*(1.0/euler_physics_double.gamm1+0.5*euler_physics_double.mach_inf_sqr);
                    } 
                    else
                    {
                        error_string = "L2_entropy_error";
                        const double entropy_inf = euler_physics_double.entropy_inf;
                        unumerical = euler_physics_double.compute_entropy_measure(soln_at_q);
                        uexact = entropy_inf;
                    }
                    l2error += pow(unumerical - uexact, 2) * fe_values_extra.JxW(iquad);
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
            last_error = l2error_mpi_sum;


            // Convergence table
            double dx = 1.0/pow(n_dofs,(1.0/dim));
            //dx = dealii::GridTools::maximal_cell_diameter(*grid);
            grid_size[igrid] = dx;
            error[igrid] = l2error_mpi_sum;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
           // convergence_table.add_value("L2_error", l2error_mpi_sum);
            convergence_table.add_value(error_string, l2error_mpi_sum);
            convergence_table.add_value("Residual",ode_solver->residual_norm);
            
            if(ode_solver->residual_norm > 1e-10)
            {
                has_residual_converged = false;
            }

            artificial_dissipation_max_coeff = dg->max_artificial_dissipation_coeff;

            pcout << " Grid size h: " << dx 
                 << " L2-error: " << l2error_mpi_sum
                 << " Residual: " << ode_solver->residual_norm
                 << std::endl;

            if (igrid > 0) {
                const double slope_soln_err = log(error[igrid]/error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     <<" " << error_string << 1 << "  " << error[igrid-1]
                     <<" " << error_string << 2 << "  " << error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
            }

            //output_results (igrid);
            if (igrid == n_grids-1)
            {
                if (param.mesh_adaptation_param.total_refinement_steps > 0)
                {
                    dealii::Point<dim> smallest_cell_coord = dg->high_order_grid->smallest_cell_coordinates();
                    pcout<<" x = "<<smallest_cell_coord[0]<<" y = "<<smallest_cell_coord[1]<<std::endl;
                    // Check if the mesh is refined near the shock i.e x \in (0.1,0.3) and y \in (0.03, 0.08).
                    if ((smallest_cell_coord[0] > 0.1) && (smallest_cell_coord[0] < 0.3) && (smallest_cell_coord[1] > 0.03) && (smallest_cell_coord[1] < 0.08)) 
                    {
                        pcout<<"Mesh is refined near the shock. Test passed"<<std::endl;
                        return 0; // Mesh adaptation test passed.
                    }
                    return 1; // Mesh adaptation failed.
                }
            }
        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates(error_string, "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific(error_string, true);
        convergence_table.set_scientific("Residual",true);
        //convergence_table.set_scientific("L2_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;
    
        double last_slope = 0.0;

        if(n_grids>=2)
        {
            last_slope = log(error[n_grids-1]/error[n_grids-2]) / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        }
        //double before_last_slope = last_slope;
        //if ( n_grids > 2 ) {
        //    before_last_slope = log(error[n_grids-2]/error[n_grids-3])
        //                        / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        //}
        //const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_avg = last_slope;
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        if(poly_degree == 0) slope_deficit_tolerance *= 2; // Otherwise, grid sizes need to be much bigger for p=0

        if( (slope_diff < slope_deficit_tolerance) || (n_grids==1) ) {
            pcout << std::endl
                 << "Convergence order not achieved. Average last 2 slopes of "
                 << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

//****************Test for artificial dissipation begins *******************************************************
    using artificial_dissipation_test_enum = Parameters::ArtificialDissipationParam::ArtificialDissipationTestType;
    artificial_dissipation_test_enum arti_dissipation_test_type = param.artificial_dissipation_param.artificial_dissipation_test_type;
    if (arti_dissipation_test_type == artificial_dissipation_test_enum::residual_convergence)
    {
        if(has_residual_converged)
        {
           pcout << std::endl << "Residual has converged. Test Passed"<<std::endl;
            return 0;
        }
        pcout << std::endl<<"Residual has not converged. Test failed" << std::endl;
        return 1;
    }
    else if (arti_dissipation_test_type == artificial_dissipation_test_enum::discontinuity_sensor_activation) 
    {
        if(artificial_dissipation_max_coeff < 1e-10)
        {
            pcout << std::endl << "Discontinuity sensor is not activated. Max dissipation coeff = " <<artificial_dissipation_max_coeff << "   Test passed"<<std::endl;
            return 0;
        }
        pcout << std::endl << "Discontinuity sensor has been activated. Max dissipation coeff = " <<artificial_dissipation_max_coeff << "   Test failed"<<std::endl;
        return 1;
    }
    else if (arti_dissipation_test_type == artificial_dissipation_test_enum::enthalpy_conservation) 
    {
        return last_error; // Return the error
    }
//****************Test for artificial dissipation ends *******************************************************
    else
    {
        int n_fail_poly = fail_conv_poly.size();
        if (n_fail_poly > 0) 
        {
            for (int ifail=0; ifail < n_fail_poly; ++ifail) 
            {
                const double expected_slope = fail_conv_poly[ifail]+1;
                const double slope_deficit_tolerance = -0.1;
                pcout << std::endl
                     << "Convergence order not achieved for polynomial p = "
                     << fail_conv_poly[ifail]
                     << ". Slope of "
                     << fail_conv_slop[ifail] << " instead of expected "
                     << expected_slope << " within a tolerance of "
                     << slope_deficit_tolerance
                     << std::endl;
            }
        }
        return n_fail_poly;
    }
}


#if PHILIP_DIM==2
    template class EulerGaussianBump <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

