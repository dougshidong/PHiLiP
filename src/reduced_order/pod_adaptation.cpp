#include "pod_adaptation.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;

template <int dim, int nstate>
PODAdaptation<dim, nstate>::PODAdaptation(std::shared_ptr<DGBase<dim,double>> &_dg, Functional<dim,nstate,double> &_functional, std::shared_ptr<ProperOrthogonalDecomposition::CoarsePOD> _coarsePOD, std::shared_ptr<ProperOrthogonalDecomposition::SpecificPOD> _finePOD)
    : functional(_functional)
    , dg(_dg)
    , coarsePOD(_coarsePOD)
    , finePOD(_finePOD)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    this->linear_solver_param.parse_parameters (parameter_handler);
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::simplePODAdaptation(int numBasisToAdd)
{
    //Compute coarse solution
    getCoarseSolution();

    //Compute fine Adjoint
    DealiiVector reducedGradient(finePOD->getPODBasis()->n());
    DealiiVector reducedAdjoint(finePOD->getPODBasis()->n());
    getReducedGradient(reducedGradient);
    applyReducedJacobianTranspose(reducedAdjoint, reducedGradient);
    DealiiVector reducedResidual(finePOD->getPODBasis()->n());

    //Project Residual to fine space
    finePOD->getPODBasis()->Tvmult(reducedResidual, dg->right_hand_side);

    //Compute dual weighted residual
    DealiiVector dualWeightedResidual(finePOD->getPODBasis()->n());
    for(unsigned int i = 0; i < reducedAdjoint.size(); i++){
        dualWeightedResidual[i] = std::abs(reducedAdjoint[i]*reducedResidual[i]);
        pcout << reducedAdjoint[i] << " " << reducedResidual[i] << " " << dualWeightedResidual[i] << std::endl;
    }

    coarsePOD->updatePODBasis(finePOD->getHighestErrorBasis(numBasisToAdd, dualWeightedResidual));

    //Re-compute POD solution with updated basis
    getCoarseSolution();

    //Compute fine Adjoint
    reducedGradient.reinit(finePOD->getPODBasis()->n());
    reducedAdjoint.reinit(finePOD->getPODBasis()->n());
    getReducedGradient(reducedGradient);
    applyReducedJacobianTranspose(reducedAdjoint, reducedGradient);
    reducedResidual.reinit(finePOD->getPODBasis()->n());

    //Project Residual to fine space
    finePOD->getPODBasis()->Tvmult(reducedResidual, dg->right_hand_side);

    //Compute dual weighted residual
    dualWeightedResidual.reinit(finePOD->getPODBasis()->n());
    for(unsigned int i = 0; i < reducedAdjoint.size(); i++){
        dualWeightedResidual[i] = reducedAdjoint[i]*reducedResidual[i];
        pcout << reducedAdjoint[i] << " " << reducedResidual[i] << " " << dualWeightedResidual[i] << std::endl;
    }
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::getCoarseSolution()
{
    ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg, coarsePOD);
    ode_solver->steady_state();
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::applyReducedJacobianTranspose(DealiiVector &reducedAdjoint, DealiiVector &reducedGradient)
{
    const bool compute_dRdW=true;
    const bool compute_dRdX=false;
    const bool compute_d2R=false;
    double flow_CFL_ = 0.0;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    dealii::TrilinosWrappers::SparseMatrix tmp;
    dealii::TrilinosWrappers::SparseMatrix reducedJacobianTranspose;
    finePOD->getPODBasis()->Tmmult(tmp, dg->system_matrix_transpose); //tmp = pod_basis^T * dg->system_matrix_transpose
    tmp.mmult(reducedJacobianTranspose, *finePOD->getPODBasis()); // reducedJacobianTranspose= tmp*pod_basis

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

    finePOD->getPODBasis()->Tvmult(reducedGradient, functional.dIdw); // reducedGradient= (pod_basis)^T * gradient
}

template class PODAdaptation <PHILIP_DIM,1>;
template class PODAdaptation <PHILIP_DIM,2>;
template class PODAdaptation <PHILIP_DIM,3>;
template class PODAdaptation <PHILIP_DIM,4>;
template class PODAdaptation <PHILIP_DIM,5>;

}
}