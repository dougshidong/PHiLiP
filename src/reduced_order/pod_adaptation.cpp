#include "pod_adaptation.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;

template <int dim, int nstate>
PODAdaptation<dim, nstate>::PODAdaptation(std::shared_ptr<DGBase<dim,double>> &_dg, Functional<dim,nstate,double> &_functional)
        : functional(_functional)
        , dg(_dg)
        , all_parameters(dg->all_parameters)
        , coarsePOD(std::make_shared<ProperOrthogonalDecomposition::CoarsePOD>(all_parameters))
        , finePOD(std::make_shared<ProperOrthogonalDecomposition::FinePOD>(all_parameters))
        , fineNotInCoarsePOD(std::make_shared<ProperOrthogonalDecomposition::FineNotInCoarsePOD>(all_parameters))
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    this->linear_solver_param.parse_parameters (parameter_handler);
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::progressivePODAdaptation()
{
    getDualWeightedResidual();
    while(abs(error) > all_parameters->reduced_order_param.adaptation_tolerance){
        std::vector<unsigned int> newColumns = getPODBasisColumnsToAdd();
        coarsePOD->addPODBasisColumns(newColumns);
        fineNotInCoarsePOD->removePODBasisColumns(newColumns);
        getDualWeightedResidual();
    }
    pcout << "Error estimate is smaller than desired tolerance!" << std::endl;
}


template <int dim, int nstate>
void PODAdaptation<dim, nstate>::simplePODAdaptation()
{
    getDualWeightedResidual();
    if(abs(error) > all_parameters->reduced_order_param.adaptation_tolerance){
        std::vector<unsigned int> newColumns = getPODBasisColumnsToAdd();
        coarsePOD->addPODBasisColumns(newColumns);
        fineNotInCoarsePOD->removePODBasisColumns(newColumns);

        getDualWeightedResidual();
    }
    else{
        pcout << "Error estimate is smaller than desired tolerance!" << std::endl;
    }
}

template <int dim, int nstate>
std::vector<unsigned int> PODAdaptation<dim, nstate>::getPODBasisColumnsToAdd()
{
    bool consider_error_sign = true;
    std::map<double, unsigned int> dualWeightedResidualToIndex;
    std::vector<unsigned int> PODBasisColumnsToAdd;
    std::map<double, unsigned int>::iterator element;

    if(consider_error_sign){
        for(unsigned int i = 0; i < dualWeightedResidual.size(); i++){
            dualWeightedResidualToIndex.emplace(dualWeightedResidual[i], fineNotInCoarsePOD->fullBasisIndices[i]);
        }
        pcout << "here" << std::endl;
        double adaptationError = error;
        if(all_parameters->reduced_order_param.adapt_coarse_basis_constant == 0){
            while(abs(adaptationError) > all_parameters->reduced_order_param.adaptation_tolerance){
                if(adaptationError > 0){
                    element = std::prev(dualWeightedResidualToIndex.end());
                }
                else{
                    element = dualWeightedResidualToIndex.begin();
                }
                PODBasisColumnsToAdd.push_back(element->second);
                pcout << "Adding POD basis: " << element->second << std::endl;
                adaptationError = adaptationError - element->first;
                pcout << "Estimated adaptation error: " << adaptationError << std::endl;
                dualWeightedResidualToIndex.erase(element);
            }
        }
        else{
            for (unsigned int i = 0; i < all_parameters->reduced_order_param.adapt_coarse_basis_constant; i++) {
                if(abs(adaptationError) < all_parameters->reduced_order_param.adaptation_tolerance){
                    break;
                }
                if(adaptationError > 0){
                    element = std::prev(dualWeightedResidualToIndex.end());
                }
                else{
                    element = dualWeightedResidualToIndex.begin();
                }
                PODBasisColumnsToAdd.push_back(element->second);
                pcout << "Adding POD basis: " << element->second << std::endl;
                adaptationError = adaptationError - element->first;
                pcout << "Estimated adaptation error: " << adaptationError << std::endl;
                dualWeightedResidualToIndex.erase(element);
            }
        }
    }
    else{
        for(unsigned int i = 0; i < dualWeightedResidual.size(); i++){
            dualWeightedResidualToIndex.emplace(abs(dualWeightedResidual[i]), fineNotInCoarsePOD->fullBasisIndices[i]);
        }
        for (unsigned int i = 0; i < all_parameters->reduced_order_param.adapt_coarse_basis_constant; i++) {
            element = std::prev(dualWeightedResidualToIndex.end());
            PODBasisColumnsToAdd.push_back(element->second);
            pcout << "Adding POD basis: " << element->second << std::endl;
            dualWeightedResidualToIndex.erase(element);
        }
    }

    return PODBasisColumnsToAdd;
}

template <int dim, int nstate>
void PODAdaptation<dim, nstate>::getDualWeightedResidual()
{
    DealiiVector fineGradient(finePOD->getPODBasis()->n());
    DealiiVector fineAdjoint(finePOD->getPODBasis()->n());
    //DealiiVector fineNotInCoarseAdjoint(fineNotInCoarsePOD->getPODBasis()->n());
    std::vector<double> fineNotInCoarseAdjoint(fineNotInCoarsePOD->getPODBasis()->n());
    DealiiVector fineNotInCoarseResidual(fineNotInCoarsePOD->getPODBasis()->n());
    dualWeightedResidual.reinit(fineNotInCoarsePOD->getPODBasis()->n());

    //Compute coarse solution
    ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg, coarsePOD);
    ode_solver->steady_state();

    //Output coarse functional
    pcout << "Coarse functional: " << std::setprecision(15) << functional.evaluate_functional(false,false) << std::setprecision(6) << std::endl;

    //Compute fine adjoint
    getReducedGradient(fineGradient);
    applyReducedJacobianTranspose(fineAdjoint, fineGradient);

    //Extract fine not in coarse adjoint
    fineAdjoint.extract_subvector_to(fineNotInCoarsePOD->fullBasisIndices, fineNotInCoarseAdjoint);

    //Compute fine not in coarse residual
    fineNotInCoarsePOD->getPODBasis()->Tvmult(fineNotInCoarseResidual, dg->right_hand_side);

    //Compute dual weighted residual
    error = 0;
    pcout << std::setw(10) << std::left << "Index" << std::setw(20) << std::left << "Reduced Adjoint" << std::setw(20) << std::left << "Reduced Residual" << std::setw(20) << std::left << "Dual Weighted Residual" << std::endl;
    for(unsigned int i = 0; i < fineNotInCoarseAdjoint.size(); i++){
        dualWeightedResidual[i] = -(fineNotInCoarseAdjoint[i] * fineNotInCoarseResidual[i]);
        error = error + dualWeightedResidual[i];
        pcout << std::setw(10) << std::left << fineNotInCoarsePOD->fullBasisIndices[i] << std::setw(20) << std::left << fineNotInCoarseAdjoint[i] << std::setw(20) << std::left << fineNotInCoarseResidual[i] << std::setw(20) << std::left << dualWeightedResidual[i] << std::endl;
    }
    pcout << std::endl << "Total error: " << error << std::endl;
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

    solve_linear (reducedJacobianTranspose, reducedGradient*=-1.0, reducedAdjoint, linear_solver_param);
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

template <int dim, int nstate>
double PODAdaptation<dim, nstate>::getCoarseFunctional()
{
    return functional.evaluate_functional(false,false);
}

template class PODAdaptation <PHILIP_DIM,1>;
template class PODAdaptation <PHILIP_DIM,2>;
template class PODAdaptation <PHILIP_DIM,3>;
template class PODAdaptation <PHILIP_DIM,4>;
template class PODAdaptation <PHILIP_DIM,5>;

}
}