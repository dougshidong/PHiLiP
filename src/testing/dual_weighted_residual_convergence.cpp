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
#include <deal.II/base/function.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "dual_weighted_residual_convergence.h"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"

#include "ode_solver/ode_solver_factory.h"
#include "mesh/mesh_adaptation.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
DualWeightedResidualConvergence<dim, nstate> :: DualWeightedResidualConvergence(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
    {}


template <int dim, int nstate>
int DualWeightedResidualConvergence<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;
    const unsigned int n_grids       = manu_grid_conv_param.number_of_grids;
    const unsigned int initial_grid_size           = manu_grid_conv_param.initial_grid_size;
    
    //std::vector<dealii::ConvergenceTable> convergence_table_vector;
    //std::vector<int> fail_conv_poly;
    //std::vector<double> fail_conv_slop;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree)
    {
       // dealii::ConvergenceTable convergence_table;
       // std::vector<double> error(n_grids);
        //std::vector<double> dx_size(n_grids);

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) 
        {
            // Create grid.
            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                 MPI_COMM_WORLD,
                 typename dealii::Triangulation<dim>::MeshSmoothing(
                 dealii::Triangulation<dim>::smoothing_on_refinement |
                 dealii::Triangulation<dim>::smoothing_on_coarsening));

            // Currently, the domain is [0,1]
            bool colorize = true;
            dealii::GridGenerator::hyper_cube(*grid, 0, 1, colorize);
            const int steps_to_create_grid = initial_grid_size + igrid;
            grid->refine_global(steps_to_create_grid);

            std::shared_ptr< DGBase<dim, double, Triangulation> > dg
                = DGFactory<dim,double,Triangulation>::create_discontinuous_galerkin(
                 &param,
                 poly_degree,
                 poly_degree+1,
                 poly_degree,
                 grid);

            dg->allocate_system();
            // initialize the solution

            /*
            std::shared_ptr<dealii::Function<dim>> initial_conditions = std::make_shared<ZeroInitialCondition<dim,double>>(nstate);

            dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
            solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
            */
            ZeroInitialCondition<dim,double> initial_conditions(nstate);
            const auto mapping = *(dg->high_order_grid->mapping_fe_field);
            dealii::VectorTools::interpolate(mapping, dg->dof_handler, initial_conditions, dg->solution);
            //dg->solution = solution_no_ghost;
            
            // generate ODE solver
            std::shared_ptr< ODE::ODESolverBase<dim,double,Triangulation> > ode_solver = ODE::ODESolverFactory<dim,double,Triangulation>::create_ODESolver(dg);

            std::cout<<"In loop"<<std::endl;
            ode_solver->steady_state();
/*
            std::shared_ptr< DualWeightedResidualError<dim,nstate,double,Triangulation> > dual_weighted_residual = std::make_shared< DualWeightedResidualError<dim,nstate,double,Triangulation>>(dg);

            const double net_dual_weighted_residual_error = dual_weighted_residual->total_dual_weighted_residual_error(dg);

            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();

            // Convergence table
             double dx = 1.0/pow(n_dofs,(1.0/dim));

             dx_size[igrid] = dx;
             error[igrid] = net_dual_weighted_residual_error;

             convergence_table.add_value("p", poly_degree);
             convergence_table.add_value("cells", n_global_active_cells);
             convergence_table.add_value("DoFs", n_dofs);
             convergence_table.add_value("dx", dx);
             convergence_table.add_value("Dual Weighted Residual", net_dual_weighted_residual_error);
             convergence_table.add_value("Residual",ode_solver->residual_norm);
             pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                   << "Grid number: " << igrid+1 << "/" << n_grids
                   << ". Number of active cells: " << n_global_active_cells
                   << ". Number of degrees of freedom: " << n_dofs
                   << std::endl;

             if (igrid > 0) 
             {
                 const double slope_soln_err = log(error[igrid]/error[igrid-1])
                                       / log(dx_size[igrid]/dx_size[igrid-1]);
                 pcout << "From grid " << igrid-1
                      << "  to grid " << igrid
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << poly_degree
                      << std::endl
                      <<" " << "Dual Weighted Residual " << 1 << "  " << error[igrid-1]
                      <<" " << "Dual Weighted Residual "<< 2 << "  " << error[igrid]
                      << "  slope " << slope_soln_err
                      << std::endl;
             }
*/
        } // for loop of igrid
/*
        pcout << " ********************************************" << std::endl
              << " Convergence rates for p = " << poly_degree << std::endl
              << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("Dual Weighted Residual", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("Dual Weighted Residual", true);
        convergence_table.set_scientific("Residual",true);

        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
         
        convergence_table_vector.push_back(convergence_table);

        double last_slope = 0.0;

        if(n_grids>=2)
        {
            last_slope = log(error[n_grids-1]/error[n_grids-2]) / log(dx_size[n_grids-1]/dx_size[n_grids-2]);
        }

        const double expected_slope = 2.0*poly_degree+1;
        const double slope_diff = last_slope - expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        slope_deficit_tolerance *= 2; 

        if( (slope_diff < slope_deficit_tolerance) ) 
        {
            pcout << std::endl
                  << "Convergence order not achieved. Average last 2 slopes of "
                  << last_slope << " instead of expected "
                  << expected_slope << " within a tolerance of "
                  << slope_deficit_tolerance
                  << std::endl;
            fail_conv_poly.push_back(poly_degree); // Store failed poly degree and slope.
            fail_conv_slop.push_back(last_slope);
        }
*/
    } // loop of poly_degree
/*
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) 
    {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

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
    */
    return 0;
}

#if PHILIP_DIM!=1
template class DualWeightedResidualConvergence <PHILIP_DIM, 1>;
template class DualWeightedResidualConvergence <PHILIP_DIM, 2>;
template class DualWeightedResidualConvergence <PHILIP_DIM, 3>;
template class DualWeightedResidualConvergence <PHILIP_DIM, 4>;
template class DualWeightedResidualConvergence <PHILIP_DIM, 5>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
