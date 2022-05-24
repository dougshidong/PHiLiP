#ifndef PHILIP_PDAS_KKT_SYSTEM_HPP
#define PHILIP_PDAS_KKT_SYSTEM_HPP

#include "ROL_Vector.hpp"
#include "ROL_Vector_SimOpt.hpp"
#include "ROL_BoundConstraint.hpp"
#include "ROL_Constraint.hpp"
#include "ROL_Types.hpp"
#include "ROL_Secant.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_ParameterList.hpp"
#include "ROL_Objective.hpp"

namespace PHiLiP {

template<typename Real>
class PDAS_KKT_System : public ROL::LinearOperator<Real> {
    private:
        const ROL::Ptr<ROL::Objective<Real> > objective_;
        const ROL::Ptr<ROL::Constraint<Real> > equality_constraints_;
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
        const ROL::Ptr<const ROL::Vector<Real> > design_variables_;
        const ROL::Ptr<const ROL::Vector<Real> > dual_equality_;
        const ROL::Ptr<const ROL::Vector<Real> > des_plus_dual_;

        ROL::Ptr<ROL::Vector<Real> > temp_design_;
        ROL::Ptr<ROL::Vector<Real> > temp_dual_equality_;
        ROL::Ptr<ROL::Vector<Real> > temp_dual_inequality_;

        ROL::Ptr<ROL::Vector<Real> > v_;

        Real add_identity_;
        Real max_eig_estimate_;
        Real bounded_constraint_tolerance_;
        const ROL::Ptr<ROL::Secant<Real> > secant_;
        bool useSecant_;
    public:
      PDAS_KKT_System(
                const ROL::Ptr<ROL::Objective<Real> > &objective,
                const ROL::Ptr<ROL::Constraint<Real> > &equality_constraints,
                const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                const ROL::Ptr<const ROL::Vector<Real> > &design_variables,
                const ROL::Ptr<const ROL::Vector<Real> > &dual_equality,
                const ROL::Ptr<const ROL::Vector<Real> > &des_plus_dual,
                const Real add_identity,
                const Real constraint_tolerance = 0,
                const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                const bool useSecant = false );
      void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const;
};

} // namespace PHiLiP
#endif
