#include "periodic_entropy_tests.h"

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// TEST FOR ENTROPY CONSERVATION/STABILITY ON PERIODIC DOMAINS (EULER/NS)
//=========================================================

template <int dim, int nstate>
PeriodicEntropyTests<dim, nstate>::PeriodicEntropyTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
}

template <int dim, int nstate>
double PeriodicEntropyTests<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // For Euler simulations, use CFL
    this->pcout << "Using CFL condition to set time step...";
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    const double constant_time_step = this->all_param.flow_solver_param.courant_friedrich_lewy_number * approximate_grid_spacing;
    this->pcout << " dt = " << constant_time_step << std::endl;
    return constant_time_step;
}

template <int dim, int nstate>
double PeriodicEntropyTests<dim, nstate>::compute_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    const double poly_degree = this->all_param.flow_solver_param.poly_degree;
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(dg->all_parameters->use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

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

    //std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }

            std::array<double,nstate> entropy_var;
            const double density = soln_state[0];
            dealii::Tensor<1,dim,double> vel;
            double vel2 = 0.0;
            for(int idim=0; idim<dim; idim++){
                vel[idim] = soln_state[idim+1]/soln_state[0];
                vel2 += vel[idim]*vel[idim];
            }
            const double pressure = 0.4*(soln_state[nstate-1] - 0.5*density*vel2);
            const double entropy = log(pressure) - 1.4 * log(density);
            
            //pcout << pressure << " " << entropy << std::endl;        
            entropy_var[0] = (1.4-entropy)/0.4 - 0.5 * density / pressure * vel2;
            for(int idim=0; idim<dim; idim++){
                entropy_var[idim+1] = soln_state[idim+1] / pressure;
            }
            entropy_var[nstate-1] = - density / pressure;
            //pcout << entropy_var[0] << " ";

            for(int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
                                                
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
            //this->pcout << entropy_var_hat_global[0] << " ";
        }
    }

    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    //double entropy_mpi = (dealii::Utilities::MPI::sum(entropy, this->mpi_communicator));
    return entropy;
}

template <int dim, int nstate>
void PeriodicEntropyTests<dim, nstate>::compute_unsteady_data_and_write_to_table(
       const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg ,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table )
{
    const double dt = this->all_param.ode_solver_param.initial_time_step;
    int output_solution_every_n_iterations = round(this->all_param.ode_solver_param.output_solution_every_dt_time_intervals/dt);
    if (this->all_param.ode_solver_param.output_solution_every_x_steps > output_solution_every_n_iterations)
        output_solution_every_n_iterations = this->all_param.ode_solver_param.output_solution_every_x_steps;


    if ((current_iteration % output_solution_every_n_iterations) == 0){
        const double entropy = this->compute_entropy(dg);
        if (current_iteration == 0)  initial_entropy = entropy;

        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << std::setprecision(16) << current_time
                    << "    Entropy: " << entropy
                    << "    U/Uo: " << entropy/initial_entropy
                    << std::endl;
    
        unsteady_data_table->add_value("iteration", current_iteration);
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        this->add_value_to_data_table(entropy,"entropy",unsteady_data_table);
        this->add_value_to_data_table(entropy/initial_entropy,"U/Uo",unsteady_data_table);
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

}

//Only template for Euler/NS
template class PeriodicEntropyTests <PHILIP_DIM,PHILIP_DIM+2>;

} // FlowSolver namespace
} // PHiLiP namespace

