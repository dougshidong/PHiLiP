#ifndef __LIMITER_CONV_TESTS__
#define __LIMITER_CONV_TESTS__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver{

//===============================================================
// Limiter Convergence Tests (Advection, Burgers, 2D Low Density)
//===============================================================
template <int dim, int nstate>
class LimiterConvTests : public FlowSolverCaseBase<dim, nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
public:
    /// Constructor
    LimiterConvTests(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~LimiterConvTests() = default;

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

protected:
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
