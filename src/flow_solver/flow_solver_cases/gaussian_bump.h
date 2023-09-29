#ifndef __GAUSSIAN_BUMP__
#define __GAUSSIAN_BUMP__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver{

//=========================================================
// Gaussian Bump
//=========================================================
template <int dim, int nstate>
class GaussianBump : public FlowSolverCaseBase<dim, nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
public:
    /// Constructor
    explicit GaussianBump(const Parameters::AllParameters *const parameters_input);

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to set the higher order grid
    void set_higher_order_grid(std::shared_ptr <DGBase<dim, double>> dg) const override;

protected:
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
