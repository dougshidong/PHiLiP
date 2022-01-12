#include "pod_adaptation.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;

template <int dim, int nstate>
PODAdaptation<dim, nstate>::PODAdaptation(std::shared_ptr<DGBase<dim,double>> &_dg, Functional<dim,nstate,double> &_functional, std::shared_ptr<ProperOrthogonalDecomposition::POD> _pod)
: functional(_functional)
, dg(_dg)
, pod(_pod)
, mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
    flow_CFL_ = 0.0;

    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    this->linear_solver_param.parse_parameters (parameter_handler);
    this->linear_solver_param.max_iterations = 1000;
    this->linear_solver_param.restart_number = 200;
    this->linear_solver_param.linear_residual = 1e-17;
    //this->linear_solver_param.ilut_fill = 1.0;//2; 50
    this->linear_solver_param.ilut_fill = 50;
    this->linear_solver_param.ilut_drop = 1e-8;
    //this->linear_solver_param.ilut_atol = 1e-3;
    //this->linear_solver_param.ilut_rtol = 1.0+1e-2;
    this->linear_solver_param.ilut_atol = 1e-5;
    this->linear_solver_param.ilut_rtol = 1.0+1e-2;
    this->linear_solver_param.linear_solver_output = Parameters::OutputEnum::verbose;
    this->linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
    //this->linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::direct;
}
/*
template <int dim, int nstate>
void PODAdaptation<dim, nstate>::generateCoarseAndFineBasis()
{
    podCoarseBasis =
    podFineBasis =
}
*/
template <int dim, int nstate>
void PODAdaptation<dim, nstate>::dualWeightedResidual()
{
    DealiiVector reducedGradient(pod->pod_basis.n());
    DealiiVector reducedAdjoint(pod->pod_basis.n());

    getReducedGradient(reducedGradient);
    applyReducedJacobianTranspose(reducedAdjoint, reducedGradient);

    std::ofstream out_file("reduced_adjoint.txt");
    reducedAdjoint.print(out_file);

    //to be completed
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::applyReducedJacobianTranspose(DealiiVector &reducedAdjoint, DealiiVector &reducedGradient)
{
    const bool compute_dRdW=true;
    const bool compute_dRdX=false;
    const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    dealii::TrilinosWrappers::SparseMatrix tmp;
    dealii::TrilinosWrappers::SparseMatrix reducedJacobianTranspose;
    pod->pod_basis.Tmmult(tmp, dg->system_matrix_transpose); //tmp = pod_basis^T * dg->system_matrix_transpose
    tmp.mmult(reducedJacobianTranspose, pod->pod_basis); // reducedJacobianTranspose= tmp*pod_basis

    solve_linear (reducedJacobianTranspose, reducedGradient, reducedAdjoint, linear_solver_param);
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::getReducedGradient(DealiiVector &reducedGradient)
{
    functional.set_state(dg->solution);
    functional.dg->high_order_grid->volume_nodes = dg->high_order_grid->volume_nodes;

    const bool compute_dIdW = true;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    pod->pod_basis.Tvmult(reducedGradient, functional.dIdw); // reducedGradient= (pod_basis)^T * gradient
}

template class PODAdaptation <PHILIP_DIM,1>;
template class PODAdaptation <PHILIP_DIM,2>;
template class PODAdaptation <PHILIP_DIM,3>;
template class PODAdaptation <PHILIP_DIM,4>;
template class PODAdaptation <PHILIP_DIM,5>;

}
}