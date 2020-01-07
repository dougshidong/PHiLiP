#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "tests.h"
#include "grid_refinement_study.h"

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"

#include "dg/dg.h"

#include "ode_solver/ode_solver.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "grid_refinement/grid_refinement.h"
#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/size_field.h"
#include "grid_refinement/gnu_out.h"

namespace PHiLiP {
    
namespace Tests {

template <int dim, int nstate>
GridRefinementStudy<dim,nstate>::GridRefinementStudy(
    const Parameters::AllParameters *const parameters_input) :
        TestsBase::TestsBase(parameters_input){}

template <int dim, int nstate>
int GridRefinementStudy<dim,nstate>::run_test() const
{
    pcout << " Running Grid Refinement Study. " << std::endl;
    const Parameters::AllParameters param           = *(TestsBase::all_parameters);
    const Parameters::GridRefinementStudyParam grs_param = param.grid_refinement_study_param;

    using ADtype = Sacado::Fad::DFad<double>;

    const unsigned int poly_degree      = grs_param.poly_degree;
    const unsigned int poly_degree_max  = grs_param.poly_degree_max;
    const unsigned int poly_degree_grid = grs_param.poly_degree_grid;

    const unsigned int grid_size = grs_param.grid_size;

    const unsigned int num_refinements = grs_param.num_refinements;


    const double left  = grs_param.grid_left;
    const double right = grs_param.grid_right;

    // creating the physics object
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double
        = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param);
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,ADtype> > physics_adtype
        = Physics::PhysicsFactory<dim,nstate,ADtype>::create_Physics(&param);

    // for each of the runs, a seperate refinement table
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    // output for convergence figure
    PHiLiP::GridRefinement::GnuFig<double> gf;

    // start of loop for each grid refinement run
    for(unsigned int iref = 0; iref <  (num_refinements?num_refinements:1); ++iref){
        // getting the parameters for this run
        const Parameters::GridRefinementParam gr_param = grs_param.grid_refinement_param;
        
        const unsigned int refinement_steps = gr_param.refinement_steps;

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        dealii::Triangulation<dim> grid(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        dealii::parallel::distributed::Triangulation<dim> grid(
            this->mpi_communicator);//,
            // typename dealii::Triangulation<dim>::MeshSmoothing(
            //     dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_refinement));
                //dealii::Triangulation<dim>::smoothing_on_refinement |
                //dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

        // generating the mesh
        dealii::GridGenerator::subdivided_hyper_cube(grid, grid_size, left, right);
        for(auto cell = grid.begin_active(); cell != grid.end(); ++cell){
            cell->set_material_id(9002);
            for(unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                if(cell->face(face)->at_boundary())
                    cell->face(face)->set_boundary_id(1000);
        }

        // generate DG
        std::shared_ptr< DGBase<dim, double> > dg 
            = DGFactory<dim,double>::create_discontinuous_galerkin(
                &param, 
                poly_degree,
                poly_degree_max,
                poly_degree_grid,
                &grid);
        dg->allocate_system();

        // initialize the solution
        // dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // generate ODE solver
        std::shared_ptr< ODE::ODESolver<dim,double> > ode_solver
            = ODE::ODESolverFactory<dim,double>::create_ODESolver(dg);
        // ode_solver->steady_state();
        // ode_solver->initialize_steady_polynomial_ramping(poly_degree);

        // generate Functional
        std::shared_ptr< Functional<dim,nstate,double> > functional 
            = FunctionalFactory<dim,nstate,double>::create_Functional(grs_param.functional_param, dg);

        // generate Adjoint
        std::shared_ptr< Adjoint<dim,nstate,double> > adjoint 
            = std::make_shared< Adjoint<dim,nstate,double> >(dg, functional, physics_adtype);

        // generate the GridRefinement
        std::shared_ptr< GridRefinement::GridRefinementBase<dim,nstate,double> >  grid_refinement 
            = GridRefinement::GridRefinementFactory<dim,nstate,double>::create_GridRefinement(grs_param.grid_refinement_param,adjoint,physics_double);

        // starting the iterations
        dealii::ConvergenceTable convergence_table;
        dealii::Vector<float> estimated_error_per_cell(grid.n_active_cells());
        
        // for plotting the error convergence with gnuplot
        std::vector<double> error;
        std::vector<double> dofs;

        for(unsigned int igrid = 0; igrid < refinement_steps; ++igrid){
            if(igrid > 0){
                grid_refinement->refine_grid();
            }

            // outputting the grid information
            const unsigned int n_global_active_cells = grid.n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << refinement_steps
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // solving the system
            // option of whether to solve the problem or interpolate it from the manufactured solution
            if(true){
                ode_solver->steady_state();
            }else{
                dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
                solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
                dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
                dg->solution = solution_no_ghost;
            }

            // TODO: computing necessary parameters
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->max_degree+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid.mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double linf_norm = 0.0;
            double l2_norm = 0.0;

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            for(auto cell = dg->dof_handler.begin_active(); cell < dg->dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                fe_values_extra.reinit(cell);
                cell->get_dof_indices(dofs_indices);

                double cell_l2error = 0.0;
                std::array<double,nstate> cell_linf;
                std::fill(cell_linf.begin(), cell_linf.end(), 0);

                for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for(unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof){
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    for(unsigned int istate = 0; istate < nstate; ++ istate){
                        const double uexact = physics_double->manufactured_solution_function->value(qpoint, istate);
                        cell_l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                        cell_linf[istate] = std::max(cell_linf[istate], abs(soln_at_q[istate]-uexact));
                    }
                }

                l2_norm += cell_l2error;
                for(unsigned int istate = 0; istate < nstate; ++ istate){
                    linf_norm = std::max(linf_norm, cell_linf[istate]);
                }
            }
            const double l2_norm_mpi = std::sqrt(dealii::Utilities::MPI::sum(l2_norm, mpi_communicator));
            const double linf_norm_mpi = dealii::Utilities::MPI::max(linf_norm, mpi_communicator);

            // computing the functional value
            double functional_value = functional->evaluate_functional(*(physics_adtype));

            // reinitializing the adjoint
            adjoint->reinit();

            bool less_than_max = dg->get_max_fe_degree() + 1 > dg->max_degree;

            // evaluating the derivatives and the fine grid adjoint
            if(less_than_max){ // don't output if at max order (as p-enrichment will segfault)
                adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double>::AdjointStateEnum::fine);
                adjoint->fine_grid_adjoint();
                estimated_error_per_cell.reinit(grid.n_active_cells());
                estimated_error_per_cell = adjoint->dual_weighted_residual();
                adjoint->output_results_vtk(igrid);
            }

            // and for the coarse grid
            adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double>::AdjointStateEnum::coarse); // this one is necessary though
            adjoint->coarse_grid_adjoint();
            adjoint->output_results_vtk(igrid);

            // convergence table
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            // convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            // convergence_table.add_value("output_error", );
            convergence_table.add_value("value", functional_value);
            convergence_table.add_value("l2_error", l2_norm_mpi);
            convergence_table.add_value("linf_error", linf_norm_mpi);
        
            error.push_back(l2_norm_mpi);
            dofs.push_back(n_dofs);
        }

        pcout << " ********************************************" << std::endl
              << " Convergence rates for p = " << poly_degree << std::endl
              << " ********************************************" << std::endl;
        // convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.set_scientific("soln_L2_error", true);
        // convergence_table.set_scientific("output_error", true);
        convergence_table.evaluate_convergence_rates("value", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("value", true);
        convergence_table.evaluate_convergence_rates("l2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("l2_error", true);
        convergence_table.evaluate_convergence_rates("linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("linf_error", true);
        if(pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);
    
        if(pcout.is_active()) 
            gf.add_xy_data(dofs, error, "l2error_"+dealii::Utilities::int_to_string(iref,1));
    }

    pcout << std::endl << std::endl << std::endl << std::endl
          << " ********************************************" << std::endl
          << " Convergence summary" << std::endl
          << " ********************************************" << std::endl;
    for(auto conv = convergence_table_vector.begin(); conv != convergence_table_vector.end(); ++conv){
        if(pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    if(pcout.is_active()){
        // formatting for the figure and outputting .gp
        gf.set_name("ErrorPlot");
        gf.set_title("Error Convergence, |u|_2 vs. Dofs");
        gf.set_x_label("# Dofs");
        gf.set_y_label("L2 Error, |u|_2");
        gf.set_grid(false);
        gf.set_x_scale_log(true);
        gf.set_y_scale_log(true);
        gf.set_legend(true);
        gf.write_gnuplot();

        // performing plotting
        gf.exec_gnuplot();
    }

    return 0;
}

template class GridRefinementStudy <PHILIP_DIM,1>;
template class GridRefinementStudy <PHILIP_DIM,2>;
template class GridRefinementStudy <PHILIP_DIM,3>;
template class GridRefinementStudy <PHILIP_DIM,4>;
template class GridRefinementStudy <PHILIP_DIM,5>;

} // namespace Tests

} // namespace PHiLiP