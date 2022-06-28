#ifndef PHILIP_EULER_BUMP_H
#define PHILIP_EULER_BUMP_H

#include "dg/dg.h"
#include "flow_solver_case_base.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include "testing/tests.h"

namespace PHiLiP {
namespace Tests{
//=========================================================
// Euler Gaussian Bump
//=========================================================

template <int dim, int nstate>
class GaussianBump : public FlowSolverCaseBase<dim, nstate>, public TestsBase{
public:
    /// Constructor
    GaussianBump(const Parameters::AllParameters *const parameters_input);
    /// Destructor
    ~GaussianBump() {}

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;
//
//    /// Display grid parameters
//    void display_grid_parameters() const;
};
}
}
#endif //PHILIP_EULER_BUMP_H
