#include "runge_kutta_base.h"

namespace PHiLiP {
namespace ODE {

template<int dim, int nspecies, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, nspecies, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input,
            std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
            : ODESolverBase<dim,nspecies,real,MeshType>(dg_input, pod)
            , relaxation_runge_kutta(RRK_object_input)
            , solver(dg_input)
{}            

template<int dim, int nspecies, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, nspecies, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
            : RungeKuttaBase(dg_input, RRK_object_input, nullptr)
{}
template<int dim, int nspecies, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, nspecies, real, n_rk_stages, MeshType>::step_in_time(real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n
    for (int istage = 0; istage < n_rk_stages; ++istage){
        this->calculate_stage_solution(istage, dt, pseudotime); // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(istage) <implicit>
        this->apply_limiter(dt);
        this->calculate_stage_derivative(istage, dt); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }
    dt = this->adjust_time_step(dt);
    this->sum_stages(dt, pseudotime); // u_np1 = u_n + dt* sum(k_i * b_i)
    this->dg->solution = this->solution_update; 
     // Calculate numerical entropy with FR correction. Does nothing if use has not selected param.
    this->FR_entropy_contribution_RRK_solver = relaxation_runge_kutta->compute_FR_entropy_contribution(dt, this->dg, this->rk_stage, true);
    this->apply_limiter(dt);
    ++(this->current_iteration);
    this->current_time += dt;
}
template <int dim, int nspecies, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, nspecies, real, n_rk_stages, MeshType>::apply_limiter (real dt)
{
    // Apply limiter at every RK stage
    if (this->limiter) {
        this->limiter->limit(this->dg->solution,
            this->dg->dof_handler,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->dg->high_order_grid->fe_system.tensor_degree(),
            this->dg->max_degree,
            this->dg->oneD_fe_collection_1state,
            this->dg->oneD_quadrature_collection,
            dt);
    }
}
template<int dim, int nspecies, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, nspecies, real, n_rk_stages, MeshType>::allocate_ode_system()
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

// Define a sequence of indices representing the range [1, 5], templates with n_rk_stages=9,10 is for LSRK
#define POSSIBLE_NRKSTAGES (1)(2)(3)(4)(5)(9)(10)

// Define a macro to instantiate MyTemplate for a specific index
#define INSTANTIATE_DISTRIBUTED(r, data, index) \
    template class RungeKuttaBase<PHILIP_DIM, PHILIP_SPECIES, double, index, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;

#if PHILIP_DIM!=1
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_DISTRIBUTED, _, POSSIBLE_NRKSTAGES)
#endif

#define INSTANTIATE_SHARED(r, data, index) \
    template class RungeKuttaBase<PHILIP_DIM, PHILIP_SPECIES, double, index, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_SHARED, _, POSSIBLE_NRKSTAGES)

#define INSTANTIATE_TRIA(r, data, index) \
    template class RungeKuttaBase<PHILIP_DIM, PHILIP_SPECIES, double, index, dealii::Triangulation<PHILIP_DIM> >;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TRIA, _, POSSIBLE_NRKSTAGES)
} // ODE namespace
} // PHiLiP namespace