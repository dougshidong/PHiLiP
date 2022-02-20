#include "explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
ExplicitODESolver<dim,real,MeshType>::ExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , rk_order(this->all_parameters->rk_order) 
        {}

template <int dim, typename real, typename MeshType>
void ExplicitODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{
    const bool compute_dRdW = false;
   this->dg->assemble_residual(compute_dRdW);
  // this->current_time += dt;
   
   Parameters::ODESolverParam ode_param = ODESolverBase<dim,real,MeshType>::all_parameters->ode_solver_param;
 //  const int rk_order = ode_param.runge_kutta_order;
    if (rk_order == 1) {
        this->dg->set_current_time(this->current_time + dt);
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->update_norm = this->solution_update.l2_norm();
        if (pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->dg->solution.add(1.0,this->solution_update);
        } else {
            this->dg->solution.add(dt,this->solution_update);
        }
        this->current_time += dt;
    } else if (rk_order == 3) {
        // Stage 0
        this->rk_stage[0] = this->dg->solution;

        // Stage 1
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose) {
            this->pcout<< "Stage 1... " << std::flush;            
        }
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[1] = this->rk_stage[0];
        //this->rk_stage[1].add(dt,this->solution_update);
        if (pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->rk_stage[1].add(1.0,this->solution_update);
        } else {
            this->rk_stage[1].add(dt,this->solution_update);
        }

        // Stage 2
        this->dg->set_current_time(this->current_time + dt/2.0);
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose) {
            this->pcout<< "2... " << std::flush;
        }
        this->dg->solution = this->rk_stage[1];
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[2] = this->rk_stage[0];
        this->rk_stage[2] *= 0.75;
        this->rk_stage[2].add(0.25, this->rk_stage[1]);
        //this->rk_stage[2].add(0.25*dt, this->solution_update);
        if (pseudotime) {
            const double CFL = 0.25*dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->rk_stage[2].add(1.0,this->solution_update);
        } else {
            this->rk_stage[2].add(0.25*dt,this->solution_update);
        }

        // Stage 3
        this->dg->set_current_time(this->current_time + dt);
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose) {
            this->pcout<< "3... " << std::flush;
        }
        this->dg->solution = this->rk_stage[2];
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[3] = this->rk_stage[0];
        this->rk_stage[3] *= 1.0/3.0;
        this->rk_stage[3].add(2.0/3.0, this->rk_stage[2]);
        //this->rk_stage[3].add(2.0/3.0*dt, this->solution_update);
        if (pseudotime) {
            const double CFL = (2.0/3.0)*dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->rk_stage[3].add(1.0,this->solution_update);
        } else {
            this->rk_stage[3].add((2.0/3.0)*dt,this->solution_update);
        }

        this->dg->solution = this->rk_stage[3];
        this->current_time += dt;
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose) {
            this->pcout<< "done." << std::endl;
        }
    }
    else if (rk_order == 4) {
        // Stage 0
        this->rk_stage[0] = this->dg->solution;
        // Stage 1
        // std::cout<< "Stage 1... " << std::flush;
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[1] = this->solution_update;
        this->rk_stage[1].operator*=(dt);
        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(0.5, this->rk_stage[1]);
        // Stage 2
        // std::cout<< "2... " << std::flush;
        this->dg->set_current_time(this->current_time + dt/2.0);
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[2] = this->solution_update;
        this->rk_stage[2].operator*=(dt);
        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(0.5, this->rk_stage[2]);
        // Stage 3
        // std::cout<< "3... " << std::flush;
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[3] = this->solution_update;
        this->rk_stage[3].operator*=(dt);
        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(1.0, this->rk_stage[3]);
        // Stage 4
        // std::cout<< "4... " << std::flush;
        this->dg->set_current_time(this->current_time + dt);
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[4] = this->solution_update;
        this->rk_stage[4].operator*=(dt);

        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(1.0/6.0, this->rk_stage[1]);
        this->dg->solution.add(1.0/3.0, this->rk_stage[2]);
        this->dg->solution.add(1.0/3.0, this->rk_stage[3]);
        this->dg->solution.add(1.0/6.0, this->rk_stage[4]);
       // pcout<< "done." << std::endl;
        this->current_time += dt;
    }
    else {
        this->pcout << "Invalid runge_kutta_order." << std::endl;
        std::abort();
    }
    ++(this->current_iteration);
}
template <int dim, typename real, typename MeshType>
void ExplicitODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating inverse mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = true;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->rk_stage.resize(rk_order+1);
    for (unsigned int i=0; i<rk_order+1; i++) {
        this->rk_stage[i].reinit(this->dg->solution);
    }
}

template class ExplicitODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ExplicitODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ExplicitODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODESolver namespace
} // PHiLiP namespace
