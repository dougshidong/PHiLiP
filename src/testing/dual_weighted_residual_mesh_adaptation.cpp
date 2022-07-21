#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include "dual_weighted_residual_mesh_adaptation.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
DualWeightedResidualMeshAdaptation<dim, nstate> :: DualWeightedResidualMeshAdaptation(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int DualWeightedResidualMeshAdaptation<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    bool use_mesh_adaptation = param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0;
    
    if(!use_mesh_adaptation)
    {
        pcout<<"This test case checks mesh adaptation. However, total mesh adaptation cycles have been set to 0 in the parameters file. Aborting..."<<std::endl; 
        std::abort();
    }

    bool check_for_p_refined_cell = (param.mesh_adaptation_param.p_refine_fraction > 0);
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
   /* 
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
            const auto mapping = *(flow_solver->dg->high_order_grid->mapping_fe_field);
            dealii::VectorTools::interpolate(mapping, flow_solver->dg->dof_handler, initial_conditions, flow_solver->dg->solution);
        
            // generate ODE solver
            std::shared_ptr< ODE::ODESolverBase<dim,double,Triangulation> > ode_solver = ODE::ODESolverFactory<dim,double,Triangulation>::create_ODESolver(dg);

            ode_solver->steady_state();
        */  

        // Check location of the most refined cell
        dealii::Point<dim> refined_cell_coord = flow_solver->dg->coordinates_of_highest_refined_cell(check_for_p_refined_cell);
        pcout<<" Coordinates of the most refined cell (x,y) = ( "<<refined_cell_coord[0]<<", "<<refined_cell_coord[1]<<")"<<std::endl;
        // Check if the mesh is refined near the shock i.e x,y in [0.3,0.6].
        if ((refined_cell_coord[0] > 0.3) && (refined_cell_coord[0] < 0.6) && (refined_cell_coord[1] > 0.3) && (refined_cell_coord[1] < 0.6))
        {
            pcout<<"Mesh is refined near the shock. Test passed!"<<std::endl;
            return 0; // Mesh adaptation test passed.
        }
        else
        {
            pcout<<"Mesh Adaptation has failed."<<std::endl;
            return 1; // Mesh adaptation failed.
        }
}

#if PHILIP_DIM==2
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 1>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
