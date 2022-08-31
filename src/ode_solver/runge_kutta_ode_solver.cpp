#include "runge_kutta_ode_solver.h"
#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
RungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , solver(dg_input)
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    
    using RKMethodEnum = Parameters::ODESolverParam::RKMethodEnum;
    const RKMethodEnum rk_method = this->ode_param.runge_kutta_method;
    if (rk_method == RKMethodEnum::ssprk3_ex){
        // RKSSP3 (RK-3 Strong-Stability-Preserving)
        const double butcher_tableau_a_values[9] = {0,0,0,1.0,0,0,0.25,0.25,0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
        this->pcout << "Assigned RK method: 3rd order SSP (explicit)" << std::endl;
    } else if (rk_method == RKMethodEnum::rk4_ex){
        // Standard RK4
        const double butcher_tableau_a_values[16] = {0,0,0,0,0.5,0,0,0,0,0.5,0,0,0,0,1.0,0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[4] = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
        this->pcout << "Assigned RK method: 4th order classical RK (explicit)" << std::endl;
    } else if (rk_method == RKMethodEnum::euler_ex){
        // Explicit Euler
        const double butcher_tableau_a_values[1] = {0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[1] = {1.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
        this->pcout << "Assigned RK method: Forward Euler (explicit)" << std::endl;
    } else if (rk_method == RKMethodEnum::euler_im){
        // Implicit Euler
        const double butcher_tableau_a_values[1] = {1.0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[1] = {1.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
        this->pcout << "Assigned RK method: Implicit Euler (implicit)" << std::endl;
    } else if (rk_method == RKMethodEnum::dirk_2_im){
        // Pareschi & Russo DIRK, x = 1 - sqrt(2)/2
        // see: wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Diagonally_Implicit_Runge%E2%80%93Kutta_methods
        const double x = 0.2928932188134525; //=1-sqrt(2)/2
        const double butcher_tableau_a_values[4] = {x,0,(1-2*x),x};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[2] = {0.5, 0.5};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
        this->pcout << "Assigned RK method: 2nd-order DIRK (implicit)" << std::endl;
    } else {
        this->pcout << "Invalid RK method. Aborting..." << std::endl;
        std::abort();
    }

}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::step_in_time (real dt, const bool pseudotime)
{  
    this->solution_update = this->dg->solution; //storing u_n
    
    //calculating stages **Note that rk_stage[i] stores the RHS at a partial time-step (not solution u)
    for (int i = 0; i < n_rk_stages; ++i){

        this->rk_stage[i]=0.0; //resets all entries to zero
        
        for (int j = 0; j < i; ++j){
            if (this->butcher_tableau_a[i][j] != 0){
                this->rk_stage[i].add(this->butcher_tableau_a[i][j], this->rk_stage[j]);
            }
        } //sum(a_ij *k_j), explicit part

        
        if(pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
        }else {
            this->rk_stage[i]*=dt; 
        }//dt * sum(a_ij * k_j)
        
        this->rk_stage[i].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
       
        //implicit solve for diagonal element
        if (this->butcher_tableau_a[i][i] != 0){
            /* // AD version
            // Solve (M/dt - dRdW) / a_ii * dw = R
            // w = w + dw
            // Note - need to have assembled residual using this->dg->assemble_residual(true);
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
            solver.solve(dt*butcher_tableau_a[i][i], rk_stage[i]);
            rk_stage[i] = solver.current_solution_estimate;

        } // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(i) <implicit>
            
        this->dg->solution = this->rk_stage[i];
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(i)))

        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[i], this->dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }

    modify_time_step(dt);

    //assemble solution from stages
    for (int i = 0; i < n_rk_stages; ++i){
        if (pseudotime){
            const double CFL = butcher_tableau_b[i] * dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
            this->solution_update.add(1.0, this->rk_stage[i]);
        } else {
            this->solution_update.add(dt* this->butcher_tableau_b[i],this->rk_stage[i]); 
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
