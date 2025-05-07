#include "runge_kutta_base.h"

namespace PHiLiP {
namespace ODE {

template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input,
            std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
            : ODESolverBase<dim,real,MeshType>(dg_input, pod)
            , relaxation_runge_kutta(RRK_object_input)
            , solver(dg_input)
{}            

template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
            : RungeKuttaBase(dg_input, RRK_object_input, nullptr)
{}
template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, real, n_rk_stages, MeshType>::step_in_time(real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n
    for (int istage = 0; istage < n_rk_stages; ++istage){
        this->calculate_stage_solution(istage, dt, pseudotime); // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(istage) <implicit>
        this->apply_limiter();
        this->calculate_stage_derivative(istage, dt); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }
    dt = this->adjust_time_step(dt);
    this->sum_stages(dt, pseudotime); // u_np1 = u_n + dt* sum(k_i * b_i)
    this->dg->solution = this->solution_update; 
     // Calculate numerical entropy with FR correction. Does nothing if use has not selected param.
    this->FR_entropy_contribution_RRK_solver = relaxation_runge_kutta->compute_FR_entropy_contribution(dt, this->dg, this->rk_stage, true);
    this->apply_limiter();
    ++(this->current_iteration);
    this->current_time += dt;
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, real, n_rk_stages, MeshType>::allocate_ode_system()
{
    this->pcout << "Allocating ODE system..." << std::flush;
    this->solution_update.reinit(this->dg->right_hand_side);

    this->pcout << std::endl;
    
    this->rk_stage.resize(n_rk_stages);
    for (int istage=0; istage<n_rk_stages; ++istage) {
        this->rk_stage[istage].reinit(this->dg->solution);
    }

    this->allocate_runge_kutta_system();
}

/*
Templates with n_rk_stages > 4 are for the LSRK method
*/
template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,5, dealii::Triangulation<PHILIP_DIM> >; 
template class RungeKuttaBase<PHILIP_DIM, double,9, dealii::Triangulation<PHILIP_DIM> >; 
template class RungeKuttaBase<PHILIP_DIM, double,10, dealii::Triangulation<PHILIP_DIM> >; 
template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,5, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,9, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,10, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,5, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,9, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,10, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
} // ODE namespace
} // PHiLiP namespace