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

#include "grid_refinement/grid_refinement.h"
#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/size_field.h"

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
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    using ADtype = Sacado::Fad::DFad<double>;

    const unsigned int poly_degree      = 1;
    const unsigned int poly_degree_max  = 1;
    const unsigned int poly_degree_grid = 1+1;

    const unsigned int grid_size = 2;

    const unsigned int refinement_steps = 5;

    const double left  = 0.0;
    const double right = 1.0;

    // creating the physics object
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double
        = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param);
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,ADtype> > physics_adtype
        = Physics::PhysicsFactory<dim,nstate,ADtype>::create_Physics(&param);

    // for each of the runs, a seperate refinement table
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    // start of loop for each grid refinement run
    {
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        dealii::Triangulation<dim> grid(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        dealii::parallel::distributed::Triangulation<dim> grid(
            this->mpi_communicator,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_refinement));
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

        // // generate Functional
        // std::shared_ptr< Functional<dim,nstate,double> > functional 
        //     = std::make_shared< L2normError<dim,nstate,double> >(dg,true,false);

        // // generate Adjoint
        // Adjoint<dim,nstate,double> adjoint(
        //     dg,
        //     functional,
        //     physics_adtype);

        // generate the GridRefinement
        std::shared_ptr< GridRefinement::GridRefinementBase<dim,nstate,double> >  grid_refinement 
            = GridRefinement::GridRefinementFactory<dim,nstate,double>::create_GridRefinement(&param,dg,physics_double);

        // starting the iterations
        dealii::ConvergenceTable convergence_table;
        dealii::Vector<float> estimated_error_per_cell(grid.n_active_cells());
        for(unsigned int igrid = 0; igrid < refinement_steps; ++igrid){
            if(igrid > 0){
                // grid_refinement->refine_grid();
                dg->allocate_system();
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
            ode_solver->steady_state();

            // TODO: computing necessary parameters

            // // reinitializing the adjoint
            // adjoint.reinit();
            
            // // evaluating the derivatives and the fine grid adjoint
            // adjoint.convert_to_state(PHiLiP::Adjoint<dim,nstate,double>::AdjointStateEnum::fine);
            // adjoint.fine_grid_adjoint();
            // estimated_error_per_cell.reinit(grid.n_active_cells());
            // estimated_error_per_cell = adjoint.dual_weighted_residual();
            // adjoint.output_results_vtk(igrid);

            // // and for the coarse grid
            // adjoint.convert_to_state(PHiLiP::Adjoint<dim,nstate,double>::AdjointStateEnum::coarse); // this one is necessary though
            // adjoint.coarse_grid_adjoint();
            // adjoint.output_results_vtk(igrid);

            // convergence table
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            // convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            // convergence_table.add_value("output_error", );
        }

        pcout << " ********************************************" << std::endl
              << " Convergence rates for p = " << poly_degree << std::endl
              << " ********************************************" << std::endl;
        // convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.set_scientific("soln_L2_error", true);
        // convergence_table.set_scientific("output_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);
    }

    pcout << std::endl << std::endl << std::endl << std::endl
          << " ********************************************" << std::endl
          << " Convergence summary" << std::endl
          << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
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