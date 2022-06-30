#ifndef PHILIP_EULER_BUMP_H
#define PHILIP_EULER_BUMP_H

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver{

//=========================================================
// Gaussian Bump
//=========================================================
template <int dim, int nstate>
class GaussianBump : public FlowSolverCaseBase<dim, nstate>
{
public:
    /// Constructor
    GaussianBump(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~GaussianBump() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif //PHILIP_EULER_BUMP_H
