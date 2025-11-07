#include <deal.II/base/tensor.h>
#include <deal.II/base/convergence_table.h>

#include "stability_fr_parameter_range.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include <fstream>
#include "ode_solver/ode_solver_factory.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
StabilityFRParametersRange<dim, nstate>::StabilityFRParametersRange(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)
: GeneralRefinementStudy<dim,nstate>(parameters_input, parameter_handler_input,
        GeneralRefinementStudy<dim,nstate>::RefinementType::h)  
{}


template <int dim, int nstate>
    Parameters::AllParameters StabilityFRParametersRange<dim, nstate>::reinit_params_c_value(const Parameters::AllParameters *parameters_in, const double c_value) const
{
    PHiLiP::Parameters::AllParameters parameters = *(parameters_in);
    parameters.FR_user_specified_correction_parameter_value = c_value;

    return parameters;
}

template <int dim, int nstate>
int StabilityFRParametersRange<dim, nstate>::run_test() const
{
    this->pcout << " Running stability ESFR parameter range test. " << std::endl;
    int testfail = 0;
    
    const unsigned int nb_c_value = this->all_parameters->flow_solver_param.number_ESFR_parameter_values;
    const double c_min = this->all_parameters->flow_solver_param.ESFR_parameter_values_start;
    const double c_max = this->all_parameters->flow_solver_param.ESFR_parameter_values_end;
    const double log_c_min = std::log10(c_min);
    const double log_c_max = std::log10(c_max);
    std::vector<double> c_array(nb_c_value+1);
    // Create log space array of c_value
    for (unsigned int ic = 0; ic < nb_c_value; ic++) {
        double log_c = log_c_min + (log_c_max - log_c_min) / (nb_c_value - 1) * ic;
        c_array[ic] = std::pow(10.0, log_c);
    }

    c_array[nb_c_value] = this->all_parameters->FR_user_specified_correction_parameter_value;

    // clear output file
    if (this->pcout.is_active()){ 
        std::ofstream conv_tab_file;
        const std::string fname = "convergence_table.txt";
        conv_tab_file.open(fname);
        conv_tab_file.close();
    }

    // Loop over c_array to compute slope
    for (unsigned int ic = 0; ic < nb_c_value+1; ic++) {
        double c_value = c_array[ic];
        this->pcout << "\n\n---------------------------------------------" << std::endl;
        this->pcout << " Entering refinement loop for a new c value = " << c_value << std::endl;
        this->pcout << "---------------------------------------------" << std::endl;

        if (this->pcout.is_active()){ 
            std::ofstream conv_tab_file;
            const std::string fname = "convergence_table.txt";
            conv_tab_file.open(fname, std::ios::app);
            conv_tab_file << "Convergence for c = " << c_value << std::endl;
            conv_tab_file.close();
        }

        const Parameters::AllParameters parameters_c_loop = reinit_params_c_value(this->all_parameters, c_value);

        const double expected_order = (c_value > 0.186) ? parameters_c_loop.flow_solver_param.poly_degree : parameters_c_loop.flow_solver_param.poly_degree + 1;

        int local_testfail = this->run_refinement_study_and_write_result(&parameters_c_loop, expected_order, true);
        if (local_testfail == 1) testfail = 1;
    }

    return testfail;
}
#if PHILIP_DIM==1
    template class StabilityFRParametersRange<PHILIP_DIM,PHILIP_DIM>;
#else
    template class StabilityFRParametersRange<PHILIP_DIM,1>;
#endif
} // Tests namespace
} // PHiLiP namespace
