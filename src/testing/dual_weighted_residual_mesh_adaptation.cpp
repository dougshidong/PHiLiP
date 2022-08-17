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

#include "dual_weighted_residual_mesh_adaptation.h"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"

#include "ode_solver/ode_solver_factory.h"
#include "mesh/mesh_adaptation.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
DualWeightedResidualMeshAdaptation<dim, nstate> :: DualWeightedResidualMeshAdaptation(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
    {}

template <int dim, int nstate>
int DualWeightedResidualMeshAdaptation<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;
    bool check_for_p_refined_cell = false;
    using MeshAdaptationTypeEnum = Parameters::MeshAdaptationParam::MeshAdaptationType;
    MeshAdaptationTypeEnum mesh_adaptation_type = param.mesh_adaptation_param.mesh_adaptation_type;
    if(mesh_adaptation_type == MeshAdaptationTypeEnum::p_adaptation)
    {
        check_for_p_refined_cell = true;
    }

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;
    const unsigned int n_grids       = manu_grid_conv_param.number_of_grids;
    const unsigned int initial_grid_size           = manu_grid_conv_param.initial_grid_size;
    
    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree)
    {
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
                 poly_degree+6,
                 poly_degree,
                 grid);

            dg->allocate_system();
            InitialConditionFunction_Zero<dim,nstate,double> initial_conditions;
            const auto mapping = *(dg->high_order_grid->mapping_fe_field);
            dealii::VectorTools::interpolate(mapping, dg->dof_handler, initial_conditions, dg->solution);
            
            // generate ODE solver
            std::shared_ptr< ODE::ODESolverBase<dim,double,Triangulation> > ode_solver = ODE::ODESolverFactory<dim,double,Triangulation>::create_ODESolver(dg);

            ode_solver->steady_state();
            
            if (param.mesh_adaptation_param.total_refinement_cycles > 0)
                 {
                    dealii::Point<dim> refined_cell_coord = dg->coordinates_of_highest_refined_cell(check_for_p_refined_cell);
                    pcout<<" x = "<<refined_cell_coord[0]<<" y = "<<refined_cell_coord[1]<<std::endl;
                    // Check if the mesh is refined near the shock i.e x,y in [0.3,0.6].
                    if ((refined_cell_coord[0] > 0.3) && (refined_cell_coord[0] < 0.6) && (refined_cell_coord[1] > 0.3) && (refined_cell_coord[1] < 0.6))
                    {
                        pcout<<"Mesh is refined near the shock. Test passed"<<std::endl;
                        return 0; // Mesh adaptation test passed.
                    }
                    else
                    {
                        pcout<<"Mesh Adaptation failed"<<std::endl;
                        return 1; // Mesh adaptation failed.
                    }
                 }
        } // for loop of igrid
    } // loop of poly_degree

    return 0; // Mesh adaptation test passed.
}

#if PHILIP_DIM!=1
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 1>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 2>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 3>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 4>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 5>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
