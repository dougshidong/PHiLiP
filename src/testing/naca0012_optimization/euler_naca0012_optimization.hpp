#ifndef __EULER_NACA_OPTIMIZATION_H__
#define __EULER_NACA_OPTIMIZATION_H__

#include <deal.II/grid/manifold_lib.h>

#include "ROL_Bounds.hpp"
#include "ROL_BoundConstraint_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"
#include "ROL_Constraint_Partitioned.hpp"

#include "testing/tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include "optimization/flow_constraints.hpp"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerNACAOptimization: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerNACAOptimization () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerNACAOptimization(const Parameters::AllParameters *const parameters_input);

    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy go to 0.
     */
    int run_test () const;

private:
    /// Actual test for which the number of design variables can be inputted.
    int optimize (const unsigned int nx_ffd, const unsigned int poly_degree) const;

};

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerNACAOptimizationConstrained: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerNACAOptimizationConstrained () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerNACAOptimizationConstrained(const Parameters::AllParameters *const parameters_input);

    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy go to 0.
     */
    int run_test () const;

private:
    /// Actual test for which the number of design variables can be inputted.
    int optimize (const unsigned int nx_ffd, const unsigned int poly_degree) const;

};

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerNACADragOptimizationLiftConstrained: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerNACADragOptimizationLiftConstrained () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerNACADragOptimizationLiftConstrained(const Parameters::AllParameters *const parameters_input);

    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy go to 0.
     */
    int run_test () const;

protected:
    /// Actual test for which the number of design variables can be inputted.
    int optimize (const unsigned int nx_ffd, const unsigned int poly_degree) const;

    ROL::Ptr<ROL::Vector<double>> getDesignVariables(
        ROL::Ptr<ROL::Vector<double>> simulation_variables,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

    ROL::Ptr<ROL::Objective<double>> getObjective(
        const ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
        const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
        const ROL::Ptr<ROL::Vector<double>> simulation_variables,
        const ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

    ROL::Ptr<ROL::BoundConstraint<double>> getDesignBoundConstraint(
        ROL::Ptr<ROL::Vector<double>> simulation_variables,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

    ROL::Ptr<ROL::Constraint<double>> getEqualityConstraint(void) const;
    ROL::Ptr<ROL::Vector<double>> getEqualityMultiplier(void) const;
    std::vector<ROL::Ptr<ROL::Constraint<double>>> getInequalityConstraint(
		const std::vector<ROL::Ptr<ROL::Objective_SimOpt<double>>> constraints,
		const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
		const ROL::Ptr<ROL::Vector<double>> simulation_variables,
		const ROL::Ptr<ROL::Vector<double>> control_variables,
		const bool is_reduced_space
		) const;

    std::vector<ROL::Ptr<ROL::Vector<double>>> getInequalityMultiplier(std::vector<double>& nonlinear_inequality_targets) const;
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> getSlackBoundConstraint(
        const std::vector<double>& nonlinear_targets,
        const std::vector<double>& lower_bound_dx,
        const std::vector<double>& upper_bound_dx) const;

    int check_flow_constraints(
        const unsigned int nx_ffd,
        ROL::Ptr<FlowConstraints<dim>> flow_constraints,
        ROL::Ptr<ROL::Vector<double>> design_simulation,
        ROL::Ptr<ROL::Vector<double>> design_control,
        ROL::Ptr<ROL::Vector<double>> dual_equality_state);
    int check_objective(
        ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
        ROL::Ptr<FlowConstraints<dim>> flow_constraints,
        ROL::Ptr<ROL::Vector<double>> design_simulation,
        ROL::Ptr<ROL::Vector<double>> design_control,
        ROL::Ptr<ROL::Vector<double>> dual_equality_state);
    int check_reduced_constraint(
        const unsigned int nx_ffd,
        ROL::Ptr<ROL::Constraint<double>> reduced_constraint,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        ROL::Ptr<ROL::Vector<double>> lift_residual_dual);
};




} // Tests namespace
} // PHiLiP namespace
#endif


