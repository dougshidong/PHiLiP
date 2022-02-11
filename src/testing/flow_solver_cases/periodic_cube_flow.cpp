#include "periodic_cube_flow.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "mesh/grids/straight_periodic_cube.hpp"
#include <deal.II/base/table_handler.h>

namespace PHiLiP {

namespace Tests {
//=========================================================
// TURBULENCE IN PERIODIC CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
PeriodicCubeFlow<dim, nstate>::PeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolver<dim,nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.grid_refinement_study_param.grid_size)
        , domain_left(this->all_param.grid_refinement_study_param.grid_left)
        , domain_right(this->all_param.grid_refinement_study_param.grid_right)
        , domain_volume(pow(domain_right - domain_left, dim))
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;

    // Flow case identifiers
    is_taylor_green_vortex = (flow_type == FlowCaseEnum::taylor_green_vortex);
}

template <int dim, int nstate>
void PeriodicCubeFlow<dim,nstate>::display_flow_solver_setup() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}
    this->pcout << "- PDE Type: " << pde_string << std::endl;
    this->pcout << "- Polynomial degree: " << this->poly_degree << std::endl;
    this->pcout << "- Courant-Friedrich-Lewy number: " << this->courant_friedrich_lewy_number << std::endl;
    this->pcout << "- Final time: " << this->final_time << std::endl;

    std::string flow_type_string;
    if(is_taylor_green_vortex) {
        flow_type_string = "Taylor Green Vortex";
        this->pcout << "- Flow Case: " << flow_type_string << std::endl;
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
        this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    }
}

template <int dim, int nstate>
void PeriodicCubeFlow<dim,nstate>
::generate_grid(std::shared_ptr<Triangulation> grid) const
{
    Grids::straight_periodic_cube<dim,dealii::parallel::distributed::Triangulation<dim>>(grid, domain_left, domain_right, number_of_cells_per_direction);
    // Display the information about the grid
    this->pcout << "\n- GRID INFORMATION:" << std::endl;
    // pcout << "- - Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << domain_left << std::endl;
    this->pcout << "- - Domain right: " << domain_right << std::endl;
    this->pcout << "- - Number of cells in each direction: " << number_of_cells_per_direction << std::endl;
    this->pcout << "- - Domain volume: " << domain_volume << std::endl;
}

template <int dim, int nstate>
double PeriodicCubeFlow<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const unsigned int number_of_degrees_of_freedom = dg->dof_handler.n_dofs();
    const double approximate_grid_spacing = (domain_right-domain_left)/pow(number_of_degrees_of_freedom,(1.0/dim));
    const double constant_time_step = this->courant_friedrich_lewy_number * approximate_grid_spacing;
    return constant_time_step;
}

template<int dim, int nstate>
double PeriodicCubeFlow<dim,nstate>::integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const
{
    // Description: Returns nondimensional kinetic energy
    const double nondimensional_density = soln_at_q[0];
    double dot_product_of_nondimensional_momentum = 0.0;
    for (int d=0; d<dim; ++d) {
        dot_product_of_nondimensional_momentum += soln_at_q[d+1]*soln_at_q[d+1];
    }
    const double nondimensional_kinetic_energy = 0.5*(dot_product_of_nondimensional_momentum)/nondimensional_density;
    return nondimensional_kinetic_energy/domain_volume;
}

template<int dim, int nstate>
double PeriodicCubeFlow<dim,nstate>::integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const
{
    // Description: Returns l2 error with the initial condition function
    // Purpose: For checking the initialization
    double integrand_value = 0.0;
    for (int istate=0; istate<nstate; ++istate) {
        const double exact_soln_at_q = this->initial_condition_function->value(qpoint, istate);
        integrand_value += pow(soln_at_q[istate] - exact_soln_at_q, 2.0);
    }
    return integrand_value;
}

template<int dim, int nstate>
double PeriodicCubeFlow<dim, nstate>::integrate_over_domain(DGBase<dim, double> &dg,const std::string integrate_what) const
{
    double integral_value = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }
            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            double integrand_value = 0.0;
            if(integrate_what=="kinetic_energy") {integrand_value = integrand_kinetic_energy(soln_at_q);}
            if(integrate_what=="l2_error_initial_condition") {integrand_value = integrand_l2_error_initial_condition(soln_at_q,qpoint);}

            integral_value += integrand_value * fe_values_extra.JxW(iquad);
        }
    }
    const double integral_value_mpi_sum = dealii::Utilities::MPI::sum(integral_value, this->mpi_communicator);
    return integral_value_mpi_sum;
}

template <int dim, int nstate>
void PeriodicCubeFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const
{
    // Compute kinetic energy
    const double kinetic_energy = integrate_over_domain(*dg,"kinetic_energy");

    if(this->mpi_rank==0) {
        // Add time to table
        std::string time_string = "time";
        unsteady_data_table->add_value(time_string, current_time);
        unsteady_data_table->set_precision(time_string, 16);
        unsteady_data_table->set_scientific(time_string, true);
        // Add kinetic energy to table
        std::string kinetic_energy_string = "kinetic_energy";
        unsteady_data_table->add_value(kinetic_energy_string, kinetic_energy);
        unsteady_data_table->set_precision(kinetic_energy_string, 16);
        unsteady_data_table->set_scientific(kinetic_energy_string, true);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Energy: " << kinetic_energy
                << std::endl;

    // Abort if energy is nan
    if(std::isnan(kinetic_energy)) {
        std::cout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
        std::cout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
}

#if PHILIP_DIM==3
    template class PeriodicCubeFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

