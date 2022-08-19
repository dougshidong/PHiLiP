#include "jacobian_vector_product.h"

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JacobianVectorProduct<dim,real,MeshType>::JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
{
    //initialize storage vectors with the same parallel structure as dg->solution
    //previous_step_solution.reinit(dg->solution);
    current_solution_estimate.reinit(dg->solution);
    current_solution_estimate_residual.reinit(dg->solution);
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::reinit_for_next_Newton_iter(dealii::LinearAlgebra::distributed::Vector<double> &current_solution_estimate_input)
{
    current_solution_estimate = current_solution_estimate_input; 
    current_solution_estimate_residual = compute_unsteady_residual(current_solution_estimate);
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>:: reinit_for_next_timestep(double dt_input,
                double epsilon_input,
                dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution_input)
{
    dt = dt_input;
    epsilon = epsilon_input;
    previous_step_solution = previous_step_solution_input;
}

template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> &w, bool do_negate) const
{
    dg->solution = w;
    dg->assemble_residual();
    
    dg->global_inverse_mass_matrix.vmult(dg->solution, dg->right_hand_side);//temp = IMM*RHS

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
void JacobianVectorProduct<dim,real,MeshType>::vmult (dealii::LinearAlgebra::distributed::Vector<double> &dst,
                const dealii::LinearAlgebra::distributed::Vector<double> &src) const
{
    dst = src;
    dst *= epsilon; 
    dst += current_solution_estimate;
    dst = compute_unsteady_residual(dst);
    dst -= current_solution_estimate_residual;
    dst *= 1.0/epsilon; // dst = 1/epsilon * (R*(current_soln_estimate + epsilon*src) - R*(curr_sol_est))

}

template class JacobianVectorProduct<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


}
}
