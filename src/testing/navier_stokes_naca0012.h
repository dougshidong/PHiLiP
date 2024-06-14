#ifndef __NAVIER_STOKES_NACA0012_H__ 
#define ___NAVIER_STOKES_NACA0012_H__

#include <deal.II/grid/manifold_lib.h>

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "physics/physics.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Test to check the goal-oriented mesh adaptation locations for various manufactured solutions.
template <int dim, int nstate>
class NavierStokesNACA0012 : public TestsBase
{
public:
    /// Constructor of DualWeightedResidualConvergence.
    NavierStokesNACA0012(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;
    void set_p_degree_and_interpolate_solution(const unsigned int poly_degree, std::shared_ptr<PHiLiP::DGBase<dim, double>> &dg) const;
    void set_p_degree_and_interpolate_solution(const unsigned int poly_degree, DGBase<dim, double> &dg) const;
    //void set_p_degree_and_interpolate_solution(const unsigned int poly_degree, &dg) const;

    /// Runs the test to check the location of refined cell after performing goal-oriented mesh adaptation.
    int run_test() const;
}; 

} // Tests namespace
} // PHiLiP namespace

#endif
