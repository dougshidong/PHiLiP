#ifndef __NACA0012__
#define __NACA0012__

// for FlowSolver class:
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include "dg/dg_base.hpp"
#include "flow_solver_case_base.h"
#include "parameters/all_parameters.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/physics.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class NACA0012 : public FlowSolverCaseBase<dim,nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
public:
    /// Constructor.
    explicit NACA0012(const Parameters::AllParameters *const parameters_input);

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to set the higher order grid
    void set_higher_order_grid(std::shared_ptr <DGBase<dim, double>> dg) const override;

    /// Will compute and print lift and drag coefficients
    void steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const override;

protected:
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    using FlowSolverCaseBase<dim,nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

public:
    /// Compute lift
    double compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const;

    /// Compute drag
    double compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
