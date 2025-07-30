#include "assemble_residual_on_subset_temporal_OOA.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
AssembleResidualSubsetOOA<dim, nstate>::AssembleResidualSubsetOOA(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input),
         n_time_calculations(parameters_input->time_refinement_study_param.number_of_times_to_solve),
         refine_ratio(parameters_input->time_refinement_study_param.refinement_ratio)
{}



template <int dim, int nstate>
Parameters::AllParameters AssembleResidualSubsetOOA<dim,nstate>::reinit_params_and_refine_timestep(int refinement) const
{
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);

    parameters.flow_solver_param.final_time*=1E-3; // Set to a very small value such that flow solver only takes one step

    //For RRK, do not end at exact time because of how relaxation parameter convergence is calculated
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    if (parameters.ode_solver_param.ode_solver_type == ODESolverEnum::rrk_explicit_solver){
        parameters.flow_solver_param.end_exactly_at_final_time = false;
    }

    return parameters;
}

template <int dim, int nstate>
double AssembleResidualSubsetOOA<dim,nstate>::calculate_Lp_error_at_final_time_wrt_function(std::shared_ptr<DGBase<dim,double>> dg, const Parameters::AllParameters parameters, double final_time, int norm_p) const
{
    //generate exact solution at final time
    std::shared_ptr<ExactSolutionFunction<dim,nstate,double>> exact_solution_function;
    exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(parameters.flow_solver_param, final_time);
    this->pcout << "End time: " << final_time << std::endl;
    double Lp_error=0;
    int overintegrate = 10;

    int poly_degree = parameters.grid_refinement_study_param.poly_degree;
    dealii::Vector<double> difference_per_cell(dg->solution.size());

    dealii::VectorTools::NormType norm_type;
    if (norm_p == 1)       norm_type = dealii::VectorTools::L1_norm;
    else if (norm_p == 2)  norm_type = dealii::VectorTools::L2_norm;
    else if (norm_p<0)     norm_type = dealii::VectorTools::Linfty_norm;
    else {
        pcout << "Norm not defined. Aborting... " << std::endl;
        std::abort();
    }

    dealii::VectorTools::integrate_difference(dg->dof_handler,    
                                              dg->solution,
                                              *exact_solution_function,
                                              difference_per_cell,
                                              dealii::QGauss<dim>(poly_degree + overintegrate), //overintegrating
                                              norm_type);
    Lp_error = dealii::VectorTools::compute_global_error(*dg->triangulation,
                                                              difference_per_cell,
                                                              norm_type);
    return Lp_error;    
}

template <int dim, int nstate>
void AssembleResidualSubsetOOA<dim,nstate>::advance_to_end_time(std::shared_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver, const double end_time) const
{
    // Herein Huen's 2nd order RK mehtod is hard-coded
    // But the residual is assembled in the two parts of the domain independantly
    const double dt = flow_solver->flow_solver_case->get_constant_time_step(flow_solver->dg);
    this->pcout << dt << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> u_n(flow_solver->dg->solution); // last solution
    dealii::LinearAlgebra::distributed::Vector<double> u_tilde(flow_solver->dg->solution); // intermediate value
    dealii::LinearAlgebra::distributed::Vector<double> f_u_n(flow_solver->dg->solution); // f (last solution)
    dealii::LinearAlgebra::distributed::Vector<double> f_u_tilde(flow_solver->dg->solution); // f( intermediate value)
    int iter=0;
    u_n = flow_solver->dg->solution;
    while (flow_solver->ode_solver->current_time < end_time) {

        u_tilde = u_n;
        //Assemble on first half of domain
        flow_solver->dg->right_hand_side*=0;
        flow_solver->dg->assemble_residual(false,false,false,0.0,0); // assemble on group ID 0

        if(this->all_parameters->use_inverse_mass_on_the_fly){
            flow_solver->dg->apply_inverse_global_mass_matrix(flow_solver->dg->right_hand_side, f_u_n);
        } else{
            flow_solver->dg->global_inverse_mass_matrix.vmult(f_u_n, flow_solver->dg->right_hand_side);
        }
        //flow_solver->dg->update_ghost_values(); // This may be needed for MPI.

        //Add first part of the RHS
        u_tilde.add(dt,f_u_n);

        // f_u_n changes in the next residual so we need to update u_n now
        u_n.add(dt/2.0,f_u_n);

        //Assemble on second half of domain
        flow_solver->dg->right_hand_side*=0;
        flow_solver->dg->assemble_residual(false,false,false,0.0,10); // assemble on group ID 10

        if(this->all_parameters->use_inverse_mass_on_the_fly){
            flow_solver->dg->apply_inverse_global_mass_matrix(flow_solver->dg->right_hand_side, f_u_n);
        } else{
            flow_solver->dg->global_inverse_mass_matrix.vmult(f_u_n, flow_solver->dg->right_hand_side);
        }

        //Add second part of the RHS
        u_tilde.add(dt,f_u_n);
        //Update u_n
        u_n.add(dt/2.0,f_u_n);

        // Store u_tilde in preparation for second stage of the RK method
        flow_solver->dg->solution = u_tilde;

        //Assemble residual on group ID 0
        flow_solver->dg->right_hand_side*=0;
        flow_solver->dg->assemble_residual(false,false,false,0.0,0); // assemble on group ID 0

        if(this->all_parameters->use_inverse_mass_on_the_fly){
            flow_solver->dg->apply_inverse_global_mass_matrix(flow_solver->dg->right_hand_side, f_u_tilde);
        } else{
            flow_solver->dg->global_inverse_mass_matrix.vmult(f_u_tilde, flow_solver->dg->right_hand_side);
        }
        //Update solution with the partially-assembled residual
        u_n.add(dt/2.0,f_u_tilde);

        // Assemble residual on group ID 10
        flow_solver->dg->right_hand_side*=0;
        flow_solver->dg->assemble_residual(false,false,false,0.0,10); // assemble on group ID 10

        if(this->all_parameters->use_inverse_mass_on_the_fly){
            flow_solver->dg->apply_inverse_global_mass_matrix(flow_solver->dg->right_hand_side, f_u_tilde);
        } else{
            flow_solver->dg->global_inverse_mass_matrix.vmult(f_u_tilde, flow_solver->dg->right_hand_side);
        }
        //Update solution with other half of the residual
        u_n.add(dt/2.0,f_u_tilde);

        flow_solver->dg->solution=u_n; 

        flow_solver->ode_solver->current_time+=dt;
        iter++;
        // Uncomment to output files
        // flow_solver->dg->output_results_vtk(iter,flow_solver->ode_solver->current_time);

    }

}

template <int dim, int nstate>
int AssembleResidualSubsetOOA<dim, nstate>::run_test() const
{
    
    const double final_time = this->all_parameters->flow_solver_param.final_time;
    const double initial_time_step = this->all_parameters->ode_solver_param.initial_time_step;
    const int n_steps = floor(final_time/initial_time_step);
    if (n_steps * initial_time_step != final_time){
        pcout << "WARNING: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, initial_time_step)
              << ". Consider modifying parameters." << std::endl;
    }

    int testfail = 0;

    dealii::ConvergenceTable convergence_table;
    double L2_error_old = 0;
    double L2_error_conv_rate=0;

    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------" << std::endl;
        pcout << "Refinement number " << refinement << " of " << n_time_calculations - 1 << std::endl;
        pcout << "---------------------------------------------" << std::endl;

        //####### Change to do only first step through flow solver
        const Parameters::AllParameters params = reinit_params_and_refine_timestep(refinement);
        std::shared_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = 
                std::move(FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler));
        static_cast<void>(flow_solver->run());
        
        pcout << "Finished first step with flowsolver " << std::endl;

        // Set half of domain to have a different group ID

        // Choose locations on which to evaluate the residual
        dealii::LinearAlgebra::distributed::Vector<int> locations_to_evaluate_rhs;
        locations_to_evaluate_rhs.reinit(flow_solver->dg->triangulation->n_active_cells());
        const auto mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
        dealii::hp::MappingCollection<dim> mapping_collection(mapping);
        dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, flow_solver->dg->fe_collection, flow_solver->dg->volume_quadrature_collection,
                                    dealii::update_quadrature_points);

        // Cell loop for assigning the locations to evaluate RHS.
        for (auto soln_cell = flow_solver->dg->dof_handler.begin_active(); soln_cell != flow_solver->dg->dof_handler.end(); ++soln_cell) {
            // First, check whether the cell is known to the current processor, either local or ghost.
            if (soln_cell->is_locally_owned() || soln_cell->is_ghost()) {
                // Get FEValues for the current cell
                const int i_fele = soln_cell->active_fe_index();
                const int i_quad = i_fele;
                const int i_mapp = 0;
                fe_values_collection.reinit (soln_cell, i_quad, i_mapp, i_fele);
                const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();

            
                // Next, check a condition which identifies cells belonging to a group.
                //////////////////////////////////////////////////////////////////////
                // This test is aiming to test on an arbitrary partition.
                // Thus, somewhat arbitrarily partitioning based on (x<0.5) for the first point of the cell.
                const double point_x = fe_values.quadrature_point(0)[0];
                if (point_x < 0.5){
                    locations_to_evaluate_rhs(soln_cell->active_cell_index())=1;
                }
                /////////////////////////////////////////////////////////////////////
            }
        }

        // set the group ID to 10 (arbitrary choice of int)
        flow_solver->dg->set_list_of_cell_group_IDs(locations_to_evaluate_rhs, 10); 
        pcout << "Assigned group ID of half of the domain." << std::endl;

        //Start time loop of hard-coded partitioned domain
        advance_to_end_time(flow_solver, this->all_parameters->flow_solver_param.final_time);

        const double final_time_actual = flow_solver->ode_solver->current_time;
        
        //check Lp error
        const double L1_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, 1);
        const double L2_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, 2);
        const double Linfty_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, -1);
        pcout << "Computed errors are: " << std::endl
              << "    L1:      " << L1_error << std::endl
              << "    L2:      " << L2_error << std::endl
              << "    Linfty:  " << Linfty_error << std::endl;

        const double dt =  params.ode_solver_param.initial_time_step;
        const int n_timesteps= flow_solver->ode_solver->current_iteration;
        pcout << " at dt = " << dt << std::endl;


        if (this->all_parameters->ode_solver_param.runge_kutta_method == PHiLiP::Parameters::ODESolverParam::RK3_2_5F_3SStarPlus 
            && this->all_parameters->ode_solver_param.atol == 1e-4 && this->all_parameters->ode_solver_param.rtol == 1e-4 
            && this->all_parameters->time_refinement_study_param.number_of_times_to_solve == 1){
            double L2_error_expected = 2.14808703658e-5; 
            pcout << " Expected L2 error is: " << L2_error_expected << std::endl;
            if (L2_error > L2_error_expected + 1e-14 || L2_error < L2_error_expected - 1e-14){
                testfail = 1;
                pcout << "Expected L2 error for RK3(2)5F[3S*+] using an atol = rtol = 1e-4 was not reached " << refinement <<std::endl;
            }
        }
        
        convergence_table.add_value("refinement", refinement);
        convergence_table.add_value("dt", dt );
        convergence_table.set_precision("dt", 16);
        convergence_table.set_scientific("dt", true);
        convergence_table.add_value("L1_error",L1_error);
        convergence_table.set_precision("L1_error", 16);
        convergence_table.evaluate_convergence_rates("L1_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        convergence_table.add_value("L2_error",L2_error);
        convergence_table.set_precision("L2_error", 16);
        convergence_table.evaluate_convergence_rates("L2_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        convergence_table.add_value("Linfty_error",Linfty_error);
        convergence_table.set_precision("Linfty_error", 16);
        convergence_table.evaluate_convergence_rates("Linfty_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        
        using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
        if(params.ode_solver_param.ode_solver_type == ODESolverEnum::rrk_explicit_solver){
            //for burgers, this is the average gamma over the runtime
            const double gamma_aggregate_m1 = final_time_actual / (n_timesteps * dt)-1;
            convergence_table.add_value("gamma_aggregate_m1", gamma_aggregate_m1);
            convergence_table.set_precision("gamma_aggregate_m1", 16);
            convergence_table.set_scientific("gamma_aggregate_m1", true);
            convergence_table.evaluate_convergence_rates("gamma_aggregate_m1", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        }
 
        //Checking convergence order
        const double expected_order = 2; // Hard-coded to order = 2
        const double order_tolerance = 0.1;
        if (refinement > 0) {
            L2_error_conv_rate = -log(L2_error_old/L2_error)/log(refine_ratio);
            pcout << "L2 order at " << refinement << " is " << L2_error_conv_rate << std::endl;
            if (abs(L2_error_conv_rate - expected_order) > order_tolerance){
                testfail = 1;
                pcout << "Expected convergence order was not reached at refinement " << refinement <<std::endl;
            }
           
            // output current time refinement results to console 
            if (refinement < n_time_calculations-1 && pcout.is_active()) convergence_table.write_text(pcout.get_stream());
        }
        L2_error_old = L2_error;
    }

    //Printing and writing convergence table
    pcout << std::endl;
    if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

    std::ofstream conv_tab_file;
    const std::string fname = "temporal_convergence_table.txt";
    conv_tab_file.open(fname);
    convergence_table.write_text(conv_tab_file);
    conv_tab_file.close();

    return testfail;
}

#if PHILIP_DIM==1
    template class AssembleResidualSubsetOOA<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
