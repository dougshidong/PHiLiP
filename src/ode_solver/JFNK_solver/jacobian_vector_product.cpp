#include "jacobian_vector_product.h"

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JacobianVectorProduct<dim,real,MeshType>::JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
{}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>:: reinit_for_next_timestep(const double dt_input,
                const double fd_perturbation_input,
                const dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution_input)
{
    dt = dt_input;
    fd_perturbation = fd_perturbation_input;
    previous_step_solution = previous_step_solution_input;
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::reinit_for_next_Newton_iter(const dealii::LinearAlgebra::distributed::Vector<double> &current_solution_estimate_input)
{
    current_solution_estimate = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(current_solution_estimate_input);
    current_solution_estimate_residual = compute_unsteady_residual(*current_solution_estimate);
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::compute_dg_residual(dealii::LinearAlgebra::distributed::Vector<double> &dst, const dealii::LinearAlgebra::distributed::Vector<double> &w) const
{
    dg->solution = w;
    dg->assemble_residual();
    
    dg->global_inverse_mass_matrix.vmult(dg->solution, dg->right_hand_side);//dg->solution = IMM * RHS
    dst = dg->solution;
}

template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_unsteady_residual(const dealii::LinearAlgebra::distributed::Vector<double> &w, 
        const bool do_negate) const
{
    compute_dg_residual(dg->solution, w); //assign residual to dg->solution

    dg->solution*=-1;

    dg->solution.add(-1.0/dt, previous_step_solution);
    dg->solution.add(1.0/dt, w);

    if (do_negate) { 
        // this is included so that -R*(w) can be found with the same
        // function for the RHS of the Newton iterations 
        // and the Jacobian estimate
        // Recall  J(wk) * dwk = -R*(wk)
        dg->solution *= -1.0; 
    } 

    return dg->solution; // R* = (w-previous_step_solution)/dt - IMM*RHS
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::vmult (dealii::LinearAlgebra::distributed::Vector<double> &destination,
                const dealii::LinearAlgebra::distributed::Vector<double> &w) const
{
    destination = w;
    destination *= fd_perturbation; 
    destination += (*current_solution_estimate);
    destination = compute_unsteady_residual(destination);
    destination -= current_solution_estimate_residual;
    destination *= 1.0/fd_perturbation; // destination = 1/fd_perturbation * (R*(current_soln_estimate + fd_perturbation*src) - R*(curr_sol_est))

}

template class JacobianVectorProduct<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


}
}
