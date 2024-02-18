#ifndef __OPTIMIZATION_H__
#define __OPTIMIZATION_H__

#include <deal.II/grid/manifold_lib.h>

#include "ROL_Bounds.hpp"
#include "ROL_BoundConstraint_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"
#include "ROL_Constraint_Partitioned.hpp"

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class OptimizationSetup: public TestsBase
{
public:
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    OptimizationSetup(const Parameters::AllParameters *const parameters_input);

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
        const ROL::Ptr<ROL::Objective_SimOpt<double>> lift_objective,
        const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
        const ROL::Ptr<ROL::Vector<double>> simulation_variables,
        const ROL::Ptr<ROL::Vector<double>> control_variables,
        const double lift_target,
        const ROL::Ptr<ROL::Objective_SimOpt<double>> volume_objective,
        const bool is_reduced_space,
        const double volume_target = -1
        ) const;

    std::vector<ROL::Ptr<ROL::Vector<double>>> getInequalityMultiplier(const double volume_target = -1) const;
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> getSlackBoundConstraint(const double lift_target, const double volume_target = -1) const;

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


