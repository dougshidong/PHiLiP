#ifndef __OPTIMIZATION_CASE_H__
#define __OPTIMIZATION_CASE_H__

namespace PHiLiP {
namespace Optimization {

struct DesignAndBounds {
    ROL::Ptr<ROL::Vector<double>> design;
    ROL::Ptr<ROL::BoundConstraint<double>> bounds;
}
class OptimizationCase {

    virtual DesignAndBounds getDesignAndBounds(
        ROL::Ptr<ROL::Vector<double>> simulation_variables,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const
    {

        // Build control bounds
        struct setLower : public ROL::Elementwise::UnaryFunction<double> {
            double apply(const double &x) const {
                const double zero = 0.0;
                if(x<zero) { return -1.0*0.1; }//ROL::ROL_INF<double>(); }
                else { return zero; }
            }
        } setlower;
        struct setUpper : public ROL::Elementwise::UnaryFunction<double> {
            double apply(const double &x) const {
                const double zero = 0.0;
                if(x>zero) { return 0.1; }//ROL::ROL_INF<double>(); }
                else { return zero; }
            }
        } setupper;

        ROL::Ptr<ROL::Vector<double>> l = control_variables->clone();
        ROL::Ptr<ROL::Vector<double>> u = control_variables->clone();

        l->applyUnary(setlower);
        u->applyUnary(setupper);

        double scale = 1;
        double feasTol = 1e-8;
        ROL::Ptr<ROL::BoundConstraint<double>> control_bounds = ROL::makePtr<ROL::Bounds<double>>(l,u, scale, feasTol);

        if (is_reduced_space) {
            return DesignAndBounds { control_variables, control_bounds };
        }

        // Build simulation bounds, but deactivate them
        ROL::Ptr<ROL::BoundConstraint<double>> simulation_bounds = ROL::makePtr<ROL::BoundConstraint<double>>(*simulation_variables);
        simulation_bounds->deactivate();
        ROL::Ptr<ROL::BoundConstraint<double>> simulation_control_bounds = ROL::makePtr<ROL::BoundConstraint_SimOpt<double>> (simulation_bounds, control_bounds);

        ROL::Ptr<ROL::Vector<double>> design_simulation_control = ROL::makePtr<ROL::Vector_SimOpt<double>>(simulation_variables, control_variables);
        return DesignAndBounds { design_simulation_control, simulation_control_bounds };
    }

    ROL::Ptr<ROL::Objective<double>> getObjective(
        const ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
        const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
        const ROL::Ptr<ROL::Vector<double>> simulation_variables,
        const ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

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
}

} // namespace Optimization
} // namespace PHiLiP

#endif
