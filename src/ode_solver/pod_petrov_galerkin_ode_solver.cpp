#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/lac/la_parallel_vector.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODPetrovGalerkinODESolver<dim,real,MeshType>::PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod)
    : ImplicitODESolver<dim,real,MeshType>(dg_input)
    , pod(pod)
{}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::step_in_time (real dt, const bool /*pseudotime*/)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw
    Parameters::ODESolverParam ode_param = ODESolverBase<dim,real,MeshType>::all_parameters->ode_solver_param;

    this->dg->system_matrix *= -1.0;

    this->dg->add_mass_matrices(1.0/dt);

    if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    /* Reference for Petrov-Galerkin projection: Refer to Equation (23) in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */
    //Petrov-Galerkin projection, petrov_galerkin_basis = V^T*J^T, pod basis V, system matrix J
    //V^T*J*V*p = -V^T*R

    this->dg->system_matrix.mmult(*this->petrov_galerkin_basis, *pod->getPODBasis()); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications

    this->petrov_galerkin_basis->Tvmult(*this->reduced_rhs, this->dg->right_hand_side); // reduced_rhs = (petrov_galerkin_basis)^T * right_hand_side

    this->petrov_galerkin_basis->Tmmult(*this->reduced_lhs, *this->petrov_galerkin_basis); //reduced_lhs = petrov_galerkin_basis^T * petrov_galerkin_basis , equivalent to V^T*J^T*J*V

    solve_linear(
            *this->reduced_lhs,
            *this->reduced_rhs,
            *this->reduced_solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

    pod->getPODBasis()->vmult(this->solution_update, *this->reduced_solution_update);

    this->linesearch();

    this->update_norm = this->solution_update.l2_norm();
    ++(this->current_iteration);
}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);

    reduced_solution_update = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(pod->getPODBasis()->n());
    reduced_rhs = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(pod->getPODBasis()->n());
    petrov_galerkin_basis = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>();
    reduced_lhs = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>();
}

template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//
