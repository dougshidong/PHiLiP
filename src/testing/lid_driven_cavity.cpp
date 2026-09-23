#include <stdlib.h>
#include <iostream>
#include <deal.II/grid/grid_refinement.h>
#include "physics/manufactured_solution.h"
#include "lid_driven_cavity.hpp"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
LidDrivenCavity<dim,nspecies,nstate>::LidDrivenCavity(const Parameters::AllParameters *const parameters_input,
                                                      const dealii::ParameterHandler &parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nspecies, int nstate>
int LidDrivenCavity<dim,nspecies,nstate>
::run_test () const
{
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    // Create grid
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    for(unsigned int d=0; d<dim; ++d)
    {
        p1[d] = -1.0;
        p2[d] = 1.0;
    }
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        #if PHILIP_DIM!=1
        this->mpi_communicator
        #endif
    );
    
    dealii::GridGenerator::hyper_rectangle	( *grid, p1, p2,true );
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 3 ) {cell->face(face)->set_boundary_id (1010);} // top
                else {cell->face(face)->set_boundary_id (1001);}
            }
        }
    }
    grid->refine_global(param.flow_solver_param.number_of_mesh_refinements);
    // Create flow solver
    std::unique_ptr<FlowSolver::FlowSolver<dim,nspecies,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nspecies,nstate>::select_flow_case(&param, parameter_handler);
    std::shared_ptr<HighOrderGrid<dim,double>> cube_grid = std::make_shared<HighOrderGrid<dim,double>>(1,grid,true,true,false);
    flow_solver->dg->set_high_order_grid(cube_grid);
    flow_solver->dg->allocate_system(false,false,false);
    // Initialize solution
    Physics::Euler<dim,nspecies,nstate,double> euler_physics_double = Physics::Euler<dim, nspecies, nstate, double>(
            &param,
            param.euler_param.ref_length,
            param.euler_param.gamma_gas,
            param.euler_param.mach_inf,
            param.euler_param.angle_of_attack,
            param.euler_param.side_slip_angle);
    auto initial_condition_function = std::make_shared<InitialConditionLidDrivenCavity<dim,nspecies,nstate>>(euler_physics_double);
    SetInitialCondition<dim,nspecies,nstate,double>::set_initial_condition(initial_condition_function, flow_solver->dg, &param);
    flow_solver->dg->solution.update_ghost_values();
    
    // Test entropy stability
    const double finalTime = param.flow_solver_param.final_time;

    pcout << " number dofs " << flow_solver->dg->dof_handler.n_dofs()<<std::endl;
    pcout << "preparing to advance solution in time" << std::endl;

    flow_solver->ode_solver->current_iteration = 0;
    flow_solver->ode_solver->allocate_ode_system();

    //loop over time
    while(flow_solver->ode_solver->current_time < finalTime){
        //get timestep
        const double time_step =  param.flow_solver_param.constant_time_step;
        if(flow_solver->ode_solver->current_iteration%param.ode_solver_param.print_iteration_modulo==0)
            pcout<<"time step "<<time_step<<" current time "<<flow_solver->ode_solver->current_time<<std::endl;
        //take the minimum timestep from all processors.
        const double dt = time_step;
        //integrate in time
        flow_solver->ode_solver->step_in_time(dt, false);
        flow_solver->ode_solver->current_iteration += 1;

        //get the change in entropy
        const double current_change_entropy = compute_change_in_entropy(flow_solver->dg, param.flow_solver_param.poly_degree);
        pcout << "Change in Entropy at time " << flow_solver->ode_solver->current_time << " is " << current_change_entropy<< std::endl;
        //check if change in entropy is non-positive
        if(current_change_entropy > 1e-12){
          pcout << " Change in entropy should be negative. Test failed..." << std::endl;
          return 1;
        }
    }
    return 0;
}
template<int dim, int nspecies, int nstate>
double LidDrivenCavity<dim,nspecies, nstate>::compute_change_in_entropy(const std::shared_ptr < DGBase<dim,nspecies, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);


    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);


    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim,nspecies, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,nspecies,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nspecies,nstate,double>::create_Physics(dg->all_parameters));

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        //get solution modal coeff
        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        //interpolate solution to quadrature points
        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        //compute entropy and kinetic energy "entropy" variables at quad points
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = euler_double->compute_entropy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
                }
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        //project the enrtopy and KE var to modal coefficients
        //then write it into a global vector
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
        }
    }
    entropy_var_hat_global.update_ghost_values();;

    //evaluate the change in entropy and change in KE
    dg->assemble_residual();
    return entropy_var_hat_global * dg->right_hand_side;
}


#if PHILIP_DIM>=2 && PHILIP_SPECIES==1
    template class LidDrivenCavity <PHILIP_DIM, PHILIP_SPECIES,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


