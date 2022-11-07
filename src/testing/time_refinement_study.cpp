#include "time_refinement_study.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"
//#include "ode_solver/runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TimeRefinementStudy<dim, nstate>::TimeRefinementStudy(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input),
         n_time_calculations(parameters_input->time_refinement_study_param.number_of_times_to_solve),
         refine_ratio(parameters_input->time_refinement_study_param.refinement_ratio)
{}



template <int dim, int nstate>
Parameters::AllParameters TimeRefinementStudy<dim,nstate>::reinit_params_and_refine_timestep(int refinement) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);
     //parameters.flow_solver_param.courant_friedrich_lewy_number*= pow(refine_ratio,refinement);

     if (dim+2==nstate)     parameters.flow_solver_param.number_of_grid_elements_per_dimension *= pow(2, refinement);
     return parameters;
}

template <int dim, int nstate>
double TimeRefinementStudy<dim,nstate>::calculate_Lp_error_at_final_time_wrt_function(std::shared_ptr<DGBase<dim,double>> dg, const Parameters::AllParameters parameters, double final_time, int norm_p) const
{
    //generate exact solution at final time
    std::shared_ptr<ExactSolutionFunction<dim,nstate,double>> exact_solution_function;
    exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(parameters.flow_solver_param, final_time);
    this->pcout << "End time: " << final_time << std::endl;
    double Lp_error=0;
    int overintegrate = 10;

    if constexpr(dim+2==nstate) { 
        // For Euler, compare only density
        // deal.ii compute_global_error() does not interface simply
        // Therefore, overintegrated error calculation is coded here
        
        // Need to do MPI sum manaully, so declaring a new local error
        double Lp_error_local = 0;

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
                const int istate = 0;
                const double u_exact = exact_solution_function->value(qpoint, istate);
                if (norm_p > 0){
                    Lp_error_local += pow(abs(soln_at_q[istate] - u_exact), norm_p) * fe_values_extra.JxW(iquad);
                } else{
                    //L-infinity norm
                    Lp_error_local = std::max(abs(soln_at_q[istate])-u_exact, Lp_error_local);
                }
            }
        }

        //MPI sum
        if (norm_p > 0) {
            Lp_error= dealii::Utilities::MPI::sum(Lp_error_local, this->mpi_communicator);
            Lp_error= pow(Lp_error, 1.0/((double)norm_p));
        } else {
            // L-infinity norm
            Lp_error = dealii::Utilities::MPI::max(Lp_error_local, this->mpi_communicator);
        }
    } else{
        int poly_degree = parameters.grid_refinement_study_param.poly_degree;
        dealii::Vector<double> difference_per_cell(dg->solution.size());

        dealii::VectorTools::NormType norm_type;
        if (norm_p == 1) norm_type = dealii::VectorTools::L1_norm;
        else if (norm_p == 2) norm_type = dealii::VectorTools::L2_norm;
        else if (norm_p<0) norm_type = dealii::VectorTools::Linfty_norm;
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
    }
    return Lp_error;    
}

template <int dim, int nstate>
int TimeRefinementStudy<dim, nstate>::run_test() const
{

    const double final_time = this->all_parameters->flow_solver_param.final_time;
    const double initial_time_step = this->all_parameters->ode_solver_param.initial_time_step;
    const int n_steps = floor(final_time/initial_time_step);
    if (n_steps * initial_time_step != final_time){
        pcout << "Error: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, initial_time_step)
              << ". Modify parameters to run this test." << std::endl;
        //std::abort();
    }

    int testfail = 0;

    dealii::ConvergenceTable convergence_table;
    double L2_error_old = 0;
    double L2_error_conv_rate=0;

    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------" << std::endl;
        pcout << "Refinement number " << refinement << " of " << n_time_calculations - 1 << std::endl;
        pcout << "---------------------------------------------" << std::endl;

        const Parameters::AllParameters params = reinit_params_and_refine_timestep(refinement);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        static_cast<void>(flow_solver->run());
        
        pcout << "Finished flowsolver " << std::endl;

        const double final_time_actual = flow_solver->ode_solver->current_time;
        
        //check Lp error
        const double L1_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, 1);
        const double L2_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, 2);
        const double Linfty_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, -1);
        pcout << "Computed errors are: " << std::endl
              << "    L1:      " << L1_error << std::endl
              << "    L2:      " << L2_error << std::endl
              << "    Linfty:  " << Linfty_error << std::endl;

        double dt = params.ode_solver_param.initial_time_step;
        if constexpr(nstate==dim+2){
            //In Euler tests, the timestep is updated according to the CFL number; initial_time_step is not used.
            std::unique_ptr<FlowSolver::PeriodicEntropyTests<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicEntropyTests<dim,nstate>>(&params);
            dt = flow_solver_case->get_constant_time_step(flow_solver->dg);
        }

        pcout << " at dt = " << dt << std::endl;
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

        if constexpr(nstate == dim+2) {
            const double gamma_agg = final_time_actual / (dt * flow_solver->ode_solver->current_iteration);

            convergence_table.add_value("gamma_agg",gamma_agg-1.0);
            convergence_table.set_precision("gamma_agg", 16);
            convergence_table.evaluate_convergence_rates("gamma_agg", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        }

        //Checking convergence order
        const double expected_order = params.ode_solver_param.rk_order;
        const double order_tolerance = 0.1;
        if (refinement > 0) {
            L2_error_conv_rate = -log(L2_error_old/L2_error)/log(refine_ratio);
            pcout << "Order at " << refinement << " is " << L2_error_conv_rate << std::endl;
            if (abs(L2_error_conv_rate - expected_order) > order_tolerance){
                testfail = 1;
                pcout << "Expected convergence order was not reached at refinement " << refinement <<std::endl;
            }
        }
        L2_error_old = L2_error;
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
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
    template class TimeRefinementStudy<PHILIP_DIM,PHILIP_DIM>;
#endif
template class TimeRefinementStudy<PHILIP_DIM,PHILIP_DIM+2>;
} // Tests namespace
} // PHiLiP namespace
