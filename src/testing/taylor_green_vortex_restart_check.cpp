#include "taylor_green_vortex_restart_check.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_cube_flow.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TaylorGreenVortexRestartCheck<dim, nstate>::TaylorGreenVortexRestartCheck(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
{}

template<int dim, int nstate>
double TaylorGreenVortexRestartCheck<dim, nstate>::compare_solutions(
    DGBase<dim, double> &dg,
    const dealii::LinearAlgebra::distributed::Vector<double> solution_reference,
    const dealii::LinearAlgebra::distributed::Vector<double> solution_to_be_checked) const
{
    double a = integrate_solution_over_domain(dg, solution_reference);
    double b = integrate_solution_over_domain(dg, solution_to_be_checked);
    pcout << "---- a = " << std::setprecision(15) << a << "---- b = " << std::setprecision(15) << b << std::endl;
    double error = (b-a)/a;
    return error;
}

template<int dim, int nstate>
double TaylorGreenVortexRestartCheck<dim, nstate>::integrate_solution_over_domain(
    DGBase<dim, double> &dg,
    const dealii::LinearAlgebra::distributed::Vector<double> solution_input) const
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
                soln_at_q[istate] += solution_input[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }
            // const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            double integrand_value = 0.0;

            for (int d=0; d<dim; ++d) {
                integrand_value += soln_at_q[d+1]*soln_at_q[d+1];
            }

            integral_value += integrand_value * fe_values_extra.JxW(iquad);
        }
    }
    const double integral_value_mpi_sum = dealii::Utilities::MPI::sum(integral_value, this->mpi_communicator);
    return integral_value_mpi_sum;
}

template <int dim, int nstate>
int TaylorGreenVortexRestartCheck<dim, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(this->all_parameters);
    static_cast<void>(flow_solver->run_test());

    dealii::LinearAlgebra::distributed::Vector<double> old_solution(flow_solver->dg->solution);
    old_solution.update_ghost_values();
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(flow_solver->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    flow_solver->dg->triangulation->save("restart");

    double error_save = compare_solutions(*(flow_solver->dg),flow_solver->dg->solution,old_solution);
    pcout << "Error from compare_solutions (save) = " << std::setprecision(15) << error_save << std::endl;

    // loading the file
    flow_solver->dg->triangulation->load("restart");
    // --- after allocate_dg
    // TO DO: Read section "Note on usage with DoFHandler with hp-capabilities" and add the stuff im missing
    // ------ Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer2(flow_solver->dg->dof_handler);
    flow_solver->dg->solution.zero_out_ghosts();
    solution_transfer2.deserialize(flow_solver->dg->solution);
    flow_solver->dg->solution.update_ghost_values();

    double error_load = compare_solutions(*(flow_solver->dg),old_solution,flow_solver->dg->solution);
    pcout << "Error from compare_solutions (load) = " << std::setprecision(15) << error_load << std::endl;

    // Compute kinetic energy
    std::unique_ptr<PeriodicCubeFlow<dim, nstate>> flow_solver_case = std::make_unique<PeriodicCubeFlow<dim,nstate>>(this->all_parameters);
    const double kinetic_energy_computed = flow_solver_case->compute_kinetic_energy(*(flow_solver->dg));

    const double relative_error = abs(kinetic_energy_computed - kinetic_energy_expected)/kinetic_energy_expected;
    if (relative_error > 1.0e-10) {
        pcout << "Computed kinetic energy is not within specified tolerance with respect to expected kinetic energy." << std::endl;
        return 1;
    }
    pcout << " Test passed, computed kinetic energy is within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class TaylorGreenVortexRestartCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace