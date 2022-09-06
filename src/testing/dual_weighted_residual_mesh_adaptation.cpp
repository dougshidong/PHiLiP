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
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;
   
    bool check_for_p_refined_cell = false;
    
    using MeshAdaptationTypeEnum = Parameters::MeshAdaptationParam::MeshAdaptationType;
    MeshAdaptationTypeEnum mesh_adaptation_type = param.mesh_adaptation_param.mesh_adaptation_type;
    if(mesh_adaptation_type == MeshAdaptationTypeEnum::p_adaptation)
    {
        check_for_p_refined_cell = true;
    }

    if(!use_mesh_adaptation)
    {
        pcout<<"This test case checks mesh adaptation. However, total mesh adaptation cycles have been set to 0 in the parameters file. Aborting..."<<std::endl; 
        std::abort();
    }
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();

    // Check location of the most refined cell
    dealii::Point<dim> refined_cell_coord = flow_solver->dg->coordinates_of_highest_refined_cell(check_for_p_refined_cell);
    pcout<<" Coordinates of the most refined cell (x,y) = ("<<refined_cell_coord[0]<<", "<<refined_cell_coord[1]<<")"<<std::endl;
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
    
