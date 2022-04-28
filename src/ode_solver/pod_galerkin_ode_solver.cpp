#include "pod_galerkin_ode_solver.h"
#include <deal.II/lac/la_parallel_vector.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODGalerkinODESolver<dim,real,MeshType>::PODGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod)
        : ImplicitODESolver<dim,real,MeshType>(dg_input)
        , pod(pod)
{}

template <int dim, typename real, typename MeshType>
void PODGalerkinODESolver<dim,real,MeshType>::step_in_time (real dt, const bool /*pseudotime*/)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw

    this->dg->system_matrix *= -1.0;

    this->dg->add_mass_matrices(1.0/dt);

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    /* Reference for Galerkin projection: Refer to Equation (21) in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */
    //Galerkin projection, pod_basis = V
    //V^T*J*V*p = -V^T*R

    pod->getPODBasis()->Tvmult(*this->reduced_rhs, this->dg->right_hand_side); // reduced_rhs = (pod_basis)^T * right_hand_side

    pod->getPODBasis()->Tmmult(*this->reduced_lhs_tmp, this->dg->system_matrix); //reduced_lhs_tmp = pod_basis^T * system_matrix

    this->reduced_lhs_tmp->mmult(*this->reduced_lhs, *pod->getPODBasis()); // reduced_lhs = reduced_lhs_tmp*pod_basis

    solve_linear(
            *this->reduced_lhs,
            *this->reduced_rhs,
            *this->reduced_solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

    pod->getPODBasis()->vmult(this->solution_update, *this->reduced_solution_update);

    linesearch();
    //double step_length = 0.01;
    //this->dg->solution.add(step_length, this->solution_update);

    this->update_norm = this->solution_update.l2_norm();
    ++(this->current_iteration);
}

template <int dim, typename real, typename MeshType>
double PODGalerkinODESolver<dim,real,MeshType>::linesearch()
{
    const auto old_solution = this->dg->solution;
    double step_length = 1.0;

    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;
    const double reduction_tolerance_2 = 2.0;

    const double initial_residual = this->dg->get_residual_l2norm();

    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }
    if (iline == 0) this->CFL_factor *= 2.0;

    if (iline == maxline) {
        step_length = 1.0;
        this->pcout << " Line search failed. Will accept any valid residual less than " << reduction_tolerance_2 << " times the current " << initial_residual << "residual. " << std::endl;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }
    if (iline == maxline) {
        this->CFL_factor *= 0.5;
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        return 0.0;
    }

    return step_length;
}

template <int dim, typename real, typename MeshType>
void PODGalerkinODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);

    reduced_solution_update = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(pod->getPODBasis()->n());
    reduced_rhs = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(pod->getPODBasis()->n());
    reduced_lhs_tmp = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>();
    reduced_lhs = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>();
}

template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace