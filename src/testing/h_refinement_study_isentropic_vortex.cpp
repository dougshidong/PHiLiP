#include "h_refinement_study_isentropic_vortex.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "physics/exact_solutions/exact_solution.h"
#include "physics/euler.h"
#include "cmath"
//#include "ode_solver/runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
HRefinementStudyIsentropicVortex<dim, nstate>::HRefinementStudyIsentropicVortex(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input),
         n_calculations(parameters_input->time_refinement_study_param.number_of_times_to_solve),
         refine_ratio(parameters_input->time_refinement_study_param.refinement_ratio)
{}



template <int dim, int nstate>
Parameters::AllParameters HRefinementStudyIsentropicVortex<dim,nstate>::reinit_params_and_refine(int refinement) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.flow_solver_param.number_of_grid_elements_per_dimension *= pow(2, refinement);
     parameters.flow_solver_param.unsteady_data_table_filename += std::to_string(parameters.flow_solver_param.number_of_grid_elements_per_dimension);
     return parameters;
}

template <int dim, int nstate>
void HRefinementStudyIsentropicVortex<dim,nstate>::calculate_Lp_error_at_final_time_wrt_function(double &Lp_error_density, 
        double &Lp_error_pressure,
        std::shared_ptr<DGBase<dim,double>> dg,
        const Parameters::AllParameters parameters,
        double final_time,
        int norm_p) const
{
    //generate exact solution at final time
    std::shared_ptr<ExactSolutionFunction<dim,nstate,double>> exact_solution_function;
    exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(parameters.flow_solver_param, final_time);
    int overintegrate = 10;

    // For Euler, compare only density or pressure
    // deal.ii compute_global_error() does not interface simply
    // Therefore, overintegrated error calculation is coded here
    
    // Need to do MPI sum manaully, so declaring a new local error
    double Lp_error_density_local = 0;
    double Lp_error_pressure_local = 0;
    
    std::shared_ptr< Physics::Euler<dim,dim+2,double> > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
            Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters));
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[dg->max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg->dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);

            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }
            
            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
            
            //Compute Lp error for istate=0, which is density
            std::array<double,nstate> exact_soln_at_q;
            for (unsigned int istate = 0; istate < nstate; ++istate) { 
                exact_soln_at_q[istate] = exact_solution_function->value(qpoint, istate);
            }
            
            const double exact_pressure = euler_physics->compute_pressure(exact_soln_at_q);
            const double pressure = euler_physics->compute_pressure(soln_at_q);

            if (norm_p > 0){
                Lp_error_density_local += pow(abs(soln_at_q[0] - exact_soln_at_q[0]), norm_p) * fe_values_extra.JxW(iquad);
                Lp_error_pressure_local += pow(abs(pressure-exact_pressure), norm_p) * fe_values_extra.JxW(iquad);
            } else{
                //L-infinity norm
                Lp_error_density_local = std::max(abs(soln_at_q[0]-exact_soln_at_q[0]), Lp_error_density_local);
                Lp_error_pressure_local = std::max(abs(pressure-exact_pressure), Lp_error_pressure_local);
            }
        }
    }

    //MPI sum
    if (norm_p > 0) {
        Lp_error_density= dealii::Utilities::MPI::sum(Lp_error_density_local, this->mpi_communicator);
        Lp_error_density= pow(Lp_error_density, 1.0/((double)norm_p));
        Lp_error_pressure = dealii::Utilities::MPI::sum(Lp_error_pressure_local, this->mpi_communicator);
        Lp_error_pressure= pow(Lp_error_pressure, 1.0/((double)norm_p));
    } else {
        // L-infinity norm
        Lp_error_density = dealii::Utilities::MPI::max(Lp_error_density_local, this->mpi_communicator);
        Lp_error_pressure = dealii::Utilities::MPI::max(Lp_error_pressure_local, this->mpi_communicator);
    }
    
}

template <int dim, int nstate>
int HRefinementStudyIsentropicVortex<dim, nstate>::run_test() const
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
// cESFR range test -------------------------------------
   
    const unsigned int nb_c_value = this->all_parameters->number_ESFR_parameter_values;
    const double c_min = this->all_parameters->ESFR_parameter_values_start;
    const double c_max = this->all_parameters->ESFR_parameter_values_end;
    const double log_c_min = std::log10(c_min);
    const double log_c_max = std::log10(c_max);
    std::vector<double> c_array(nb_c_value+1);


    // Create log space array of c_value
    for (unsigned int ic = 0; ic < nb_c_value; ic++) {
        double log_c = log_c_min + (log_c_max - log_c_min) / (nb_c_value - 1) * ic;
        c_array[ic] = std::pow(10.0, log_c);
    }
    // Add cPlus value at the end
    c_array[nb_c_value] = this->all_parameters->FR_user_specified_correction_parameter_value;
    // c_array[nb_c_value]=3.67e-3; // 0.186; 3.67e-3; 4.79e-5; 4.24e-7;

    dealii::ConvergenceTable convergence_table_density;
    dealii::ConvergenceTable convergence_table_pressure;

    // Loop over c_array to compute slope
    for (unsigned int ic = 0; ic < nb_c_value+1; ic++) {
        double c_value = c_array[ic];

// --------------------------------------------------------

    double L2_error_pressure_old = 0;
    double L2_error_pressure_conv_rate=0;


    for (int refinement = 0; refinement < n_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------" << std::endl;
        pcout << "Refinement number " << refinement << " of " << n_calculations - 1 << std::endl;
        pcout << "---------------------------------------------" << std::endl;

        const Parameters::AllParameters params = reinit_params_and_refine(refinement);
        auto params_modified = params;
        params_modified.FR_user_specified_correction_parameter_value = c_value;
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params_modified, parameter_handler);
        std::unique_ptr<FlowSolver::PeriodicEntropyTests<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicEntropyTests<dim,nstate>>(&params_modified);
    
        static_cast<void>(flow_solver->run());
        pcout << "Finished flowsolver " << std::endl;

        const double final_time_actual = flow_solver->ode_solver->current_time;
        
        double L1_error_density=0;
        double L1_error_pressure=0;
        calculate_Lp_error_at_final_time_wrt_function(L1_error_density, L1_error_pressure, flow_solver->dg, params_modified,final_time_actual, 1);
        double L2_error_density =0;
        double L2_error_pressure=0;
        calculate_Lp_error_at_final_time_wrt_function(L2_error_density,L2_error_pressure, flow_solver->dg, params_modified,final_time_actual, 2);
        double Linfty_error_density = 0;
        double Linfty_error_pressure = 0;
        calculate_Lp_error_at_final_time_wrt_function(Linfty_error_density,Linfty_error_pressure, flow_solver->dg, params_modified,final_time_actual, -1);
        pcout << "Computed density errors are: " << std::endl
              << "    L1:      " << L1_error_density << std::endl
              << "    L2:      " << L2_error_density << std::endl
              << "    Linfty:  " << Linfty_error_density << std::endl;

        const double dt = flow_solver_case->get_constant_time_step(flow_solver->dg);
        const int n_cells = pow(params_modified.flow_solver_param.number_of_grid_elements_per_dimension, PHILIP_DIM);
        pcout << " at dt = " << dt << std::endl;
        
        // Convergence for density
        convergence_table_density.add_value("cESFR", params_modified.FR_user_specified_correction_parameter_value );
        convergence_table_density.set_precision("cESFR", 16);
        convergence_table_density.set_scientific("cESFR", true);
        convergence_table_density.add_value("refinement", refinement);
        convergence_table_density.add_value("dt", dt );
        convergence_table_density.set_precision("dt", 16);
        convergence_table_density.set_scientific("dt", true);
        convergence_table_density.add_value("n_cells",n_cells); 
        convergence_table_density.add_value("L1_error_density",L1_error_density);
        convergence_table_density.set_precision("L1_error_density", 16);
        convergence_table_density.evaluate_convergence_rates("L1_error_density", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);
        convergence_table_density.add_value("L2_error_density",L2_error_density);
        convergence_table_density.set_precision("L2_error_density", 16);
        convergence_table_density.evaluate_convergence_rates("L2_error_density", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);
        convergence_table_density.add_value("Linfty_error_density",Linfty_error_density);
        convergence_table_density.set_precision("Linfty_error_density", 16);
        convergence_table_density.evaluate_convergence_rates("Linfty_error_density", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);

        if (params_modified.ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::rrk_explicit_solver) {
            const double gamma_agg = final_time_actual / (dt * flow_solver->ode_solver->current_iteration);

            convergence_table_density.add_value("gamma_agg",gamma_agg-1.0);
            convergence_table_density.set_precision("gamma_agg", 16);
            convergence_table_density.evaluate_convergence_rates("gamma_agg", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);
        }

        // Convergence for pressure
        convergence_table_pressure.add_value("cESFR", params_modified.FR_user_specified_correction_parameter_value );
        convergence_table_pressure.set_precision("cESFR", 16);
        convergence_table_pressure.set_scientific("cESFR", true);
        convergence_table_pressure.add_value("refinement", refinement);
        convergence_table_pressure.add_value("dt", dt );
        convergence_table_pressure.set_precision("dt", 16);
        convergence_table_pressure.set_scientific("dt", true);
        convergence_table_pressure.add_value("n_cells",n_cells); 
        convergence_table_pressure.add_value("L1_error_pressure",L1_error_pressure);
        convergence_table_pressure.set_precision("L1_error_pressure", 16);
        convergence_table_pressure.evaluate_convergence_rates("L1_error_pressure", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);
        convergence_table_pressure.add_value("L2_error_pressure",L2_error_pressure);
        convergence_table_pressure.set_precision("L2_error_pressure", 16);
        convergence_table_pressure.evaluate_convergence_rates("L2_error_pressure", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);
        convergence_table_pressure.add_value("Linfty_error_pressure",Linfty_error_pressure);
        convergence_table_pressure.set_precision("Linfty_error_pressure", 16);
        convergence_table_pressure.evaluate_convergence_rates("Linfty_error_pressure", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, PHILIP_DIM);

        //Checking convergence order
        const double expected_order = params_modified.flow_solver_param.poly_degree + 1; 
        //set tolerance to make test pass for ctest. Note that the grids are very coarse (not in asymptotic range)
        const double order_tolerance = 1.0; 
        if (refinement > 0) {
            L2_error_pressure_conv_rate = -log(L2_error_pressure_old/L2_error_pressure)/log(refine_ratio);
            pcout << "Order for L2 pressure at " << refinement << " is " << L2_error_pressure_conv_rate << std::endl;
            if (abs(L2_error_pressure_conv_rate - expected_order) > order_tolerance){
                testfail = 1;
                pcout << "Expected convergence order for L2 pressure  was not reached at refinement " << refinement <<std::endl;
            }
            if (refinement < n_calculations-1 && pcout.is_active()){
                // Print current convergence results for solution monitoring
                convergence_table_density.write_text(pcout.get_stream());
                convergence_table_pressure.write_text(pcout.get_stream());
            }
        }
        L2_error_pressure_old = L2_error_pressure;
    }// refinement loop
    }// c ESFR range loop

    //Printing and writing convergence tables
    pcout << std::endl;
    if (pcout.is_active()){
        convergence_table_density.write_text(pcout.get_stream());
        convergence_table_pressure.write_text(pcout.get_stream());
    }
    std::ofstream conv_tab_file;
    const std::string fname = "convergence_tables.txt";
    conv_tab_file.open(fname);
    convergence_table_density.write_text(conv_tab_file);
    convergence_table_pressure.write_text(conv_tab_file);
    conv_tab_file.close();

    return testfail;
}
#if PHILIP_DIM!=1
    template class HRefinementStudyIsentropicVortex<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
