#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/lac/la_parallel_vector.h>
#include <ctime>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODPetrovGalerkinODESolver<dim,real,MeshType>::PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD> pod)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , pod(pod)
{}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::step_in_time (real dt, const bool /*pseudotime*/)
{
    double duration;
    std::clock_t start;
    start = std::clock();

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

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    this->pcout << "First section of step_in_time time: "<< duration <<'\n';

    //Petrov-Galerkin projection, basis psi = V^T*J^T
    //V^T*J*V*p = -V^T*R

    start = std::clock();

    this->dg->system_matrix.mmult(this->psi, pod->pod_basis); // psi = system_matrix * pod_basis. Note, use transpose in subsequent multiplications

    this->psi.Tvmult(this->reduced_rhs, this->dg->right_hand_side); // reduced_rhs = (psi)^T * right_hand_side

    this->psi.Tmmult(this->reduced_lhs, this->psi); //reduced_lhs = psi^T * psi , equivalent to V^T*J^T*J*V


    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    this->pcout << "Multiplication time: "<< duration <<'\n';

    start = std::clock();

    solve_linear(
            this->reduced_lhs,
            this->reduced_rhs,
            this->reduced_solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

    pod->pod_basis.vmult(this->solution_update, this->reduced_solution_update);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    this->pcout << "Solver time: "<< duration <<'\n';

    start = std::clock();

    const double initial_residual = this->dg->get_residual_l2norm();
    double step_length = 1.0;
    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual();
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    this->update_norm = this->solution_update.l2_norm();

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    this->pcout << "Last section of step_in_time time: "<< duration <<'\n';

}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);

    this->reduced_solution_update.reinit(pod->pod_basis.n());
    this->psi.reinit(dealii::SparsityPattern(pod->pod_basis.m(), pod->pod_basis.n(), pod->pod_basis.n()));
    this->reduced_rhs.reinit(pod->pod_basis.n());
    this->reduced_lhs.reinit(dealii::SparsityPattern(pod->pod_basis.n(), pod->pod_basis.n(), pod->pod_basis.n()));
}

template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//
