#include "explicit_ode_solver.h"
//#include "runge_kutta_ode_solve.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
RungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , butcher_tableau(rk_tableau_input)
        , solver(dg_input)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::step_in_time (real dt, const bool pseudotime)
{  
    this->solution_update = this->dg->solution; //storing u_n
    
    //calculating stages **Note that rk_stage[i] stores the RHS at a partial time-step (not solution u)
    for (int i = 0; i < n_rk_stages; ++i){

        this->rk_stage[i]=0.0; //resets all entries to zero
        
        for (int j = 0; j < i; ++j){
            if (this->butcher_tableau->get_a(i,j) != 0){
                this->rk_stage[i].add(this->butcher_tableau->get_a(i,j), this->rk_stage[j]);
            }
        } //sum(a_ij *k_j), explicit part

        
        if(pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
        }else {
            this->rk_stage[i]*=dt; 
        }//dt * sum(a_ij * k_j)
        
        this->rk_stage[i].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
       
        //implicit solve if there is a nonzero diagonal element
        if (!this->butcher_tableau_aii_is_zero[i]){
            /* // AD version - keeping in comments as it may be useful for future testing
            // Solve (M/dt - dRdW) / a_ii * dw = R
            // w = w + dw
            // Note - need to have assembled residual using this->dg->assemble_residual(true);
            //        and have mass matrix assembled, and include linear_solver
            dealii::LinearAlgebra::distributed::Vector<double> temp_u(this->dg->solution.size());

            this->dg->system_matrix *= -1.0/butcher_tableau_a[i][i]; //system_matrix = -1/a_ii*dRdW
            this->dg->add_mass_matrices(1.0/butcher_tableau_a[i][i]/dt); //system_matrix = -1/a_ii*dRdW + M/dt/a_ii = A

            solve_linear ( //Solve Ax=b using Aztec00 gmres
                        this->dg->system_matrix, //A = -1/a_ii*dRdW + M/dt/a_ii
                        this->dg->right_hand_side, //b = R
                        temp_u, // result,  x = dw
                        this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

            this->rk_stage[i].add(1.0, temp_u);
            */

            //JFNK version
            solver.solve(dt*this->butcher_tableau->get_a(i,i), this->rk_stage[i]);
            this->rk_stage[i] = solver.current_solution_estimate;

        } // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(i) <implicit>
            
        this->dg->solution = this->rk_stage[i];
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(i)))

        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[i], this->dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }

    modify_time_step(dt);

    //assemble solution from stages
    for (int i = 0; i < n_rk_stages; ++i){
        if (pseudotime){
            const double CFL = this->butcher_tableau->get_b(i) * dt;
            this->dg->time_scale_solution_update(this->rk_stage[i], CFL);
            this->solution_update.add(1.0, this->rk_stage[i]);
        } else {
            this->solution_update.add(dt* this->butcher_tableau->get_b(i),this->rk_stage[i]); 
        }
    }
    this->dg->solution = this->solution_update; // u_np1 = u_n + dt* sum(k_i * b_i)

    ++(this->current_iteration);
    this->current_time += dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::modify_time_step(real &/*dt*/)
{
    //do nothing
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating inverse mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = true;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->rk_stage.resize(n_rk_stages);
    for (int i=0; i<n_rk_stages; ++i) {
        this->rk_stage[i].reinit(this->dg->solution);
    }

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int i=0; i<n_rk_stages; ++i) {
        if (this->butcher_tableau->get_a(i,i)==0.0)     this->butcher_tableau_aii_is_zero[i] = true;
    }
}

template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
