#ifndef __PHILIP_CONSTRAINTFROMOBJECTIVE_SIMOPT_H__
#define __PHILIP_CONSTRAINTFROMOBJECTIVE_SIMOPT_H__
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_SingletonVector.hpp"

namespace PHiLiP {

/// Creates a constraint from an objective function and a offset value.
/** Same as ROL::ConstraintFromObjective except we do it for the SimOpt version.
 *  This will be especially useful for example when we have a target lift constraint
 *  such as CL - 0.375 = 0.
 *
 *  Note that Constraints usually provide a vector of constraints with multiple entries
 *  and that we rarely need to access 1 specific entry value.
 *  However, in the case of an objective, it only contains a single value, hence the use
 *  of the SingletonVector. (Weird naming since it has nothing to do with the typical
 *  "Singleton" design pattern).
 */
template<class Real> 
class ConstraintFromObjective_SimOpt : public ROL::Constraint_SimOpt<Real> {

    /// Shorthand for ROL::Vector.
    using V = ROL::Vector<Real>;

private:

    /// Underlying objective function.
    const ROL::Ptr<ROL::Objective_SimOpt<Real> > obj_;
    /// Vector of size n_sim to store dIdX1.
    ROL::Ptr<V>                                  dualVector_1_;
    /// Vector of size n_ctl to store dIdX2.
    ROL::Ptr<V>                                  dualVector_2_;
    /// Offset value defining the constraint on the objective function.
    const Real                                   offset_;
    /// Check whether dualVector_1_ has been initialized.
    bool                                         isDual1Initialized_;
    /// Check whether dualVector_2_ has been initialized.
    bool                                         isDual2Initialized_;

    /// Get a vector's entry value, where the vector is a SingletonVector.
    Real getValue( const V& x ) { 
        return dynamic_cast<const ROL::SingletonVector<Real>&>(x).getValue(); 
    }
 
    /// Set a value on the vector, where the vector is a SingletonVector.
    void setValue( V& x, Real val ) {
        dynamic_cast<ROL::SingletonVector<Real>&>(x).setValue(val);
    }

public:
    using ROL::Constraint_SimOpt<double>::value;
    using ROL::Constraint_SimOpt<double>::update;
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_1;
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_2;

    /// Constructor.
    ConstraintFromObjective_SimOpt(
        const ROL::Ptr<ROL::Objective_SimOpt<Real> > &obj,
        const Real offset = 0 )
        : obj_(obj)
        , dualVector_1_(ROL::nullPtr)
        , dualVector_2_(ROL::nullPtr)
        , offset_(offset)
        , isDual1Initialized_(false)
        , isDual2Initialized_(false) {}

    /// Get underlying objective function.
    const ROL::Ptr<ROL::Objective_SimOpt<Real> > getObjective(void) const { return obj_; }


    /// Set parameter on constraint.
    void setParameter( const std::vector<Real> &param ) override
    {
        obj_->setParameter(param);
        ROL::Constraint<Real>::setParameter(param);
    }


    /// Update constraint (underlying objective)'s design varibles.
    void update( const V& des_var_sim, const V& des_var_ctl, bool flag = true, int iter = -1 ) override
    {
        obj_->update(des_var_sim, des_var_ctl, flag, iter);
    }

    //void update_1( const V& des_var_sim, bool flag, int iter )
    //{
    //}

    //void update_2( const V& des_var_ctl, bool flag, int iter )
    //{
    //}
  
    /// Returns the constraint value equal to the objective value minus the offset.
    void value( V& c, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        setValue(c, obj_->value(des_var_sim, des_var_ctl, tol) - offset_ ); 
    }

    /// Returns dIdX1.dot(v) where v has the size of X1
    void applyJacobian_1( V& jacobian_vector, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        if ( !isDual1Initialized_ ) {
          dualVector_1_ = des_var_sim.dual().clone();
          isDual1Initialized_ = true;
        }
        obj_->gradient_1(*dualVector_1_, des_var_sim, des_var_ctl, tol);
        setValue(jacobian_vector, v.dot(dualVector_1_->dual()));
    }
    /// Returns dIdX2.dot(v) where v has the size of X2
    void applyJacobian_2( V& jacobian_vector, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        if ( !isDual2Initialized_ ) {
          dualVector_2_ = des_var_ctl.dual().clone();
          isDual2Initialized_ = true;
        }
        obj_->gradient_2(*dualVector_2_, des_var_sim, des_var_ctl, tol);
        setValue(jacobian_vector, v.dot(dualVector_2_->dual()));
    }

    /// Returns v*dIdX1 where v should be a scalar.
    void applyAdjointJacobian_1( V& ajv, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        obj_->gradient_1(ajv, des_var_sim, des_var_ctl, tol);
        ajv.scale(getValue(v));
    }

    /// Returns v*dIdX2 where v should be a scalar.
    void applyAdjointJacobian_2( V& ajv, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        obj_->gradient_2(ajv, des_var_sim, des_var_ctl, tol);
        ajv.scale(getValue(v));
    }

    /// Returns dual*dIdX1dX1*input_vector where dual should be a scalar, and input_vector has size X1.
    void applyAdjointHessian_11 ( ROL::Vector<double> &output_vector,
                                  const ROL::Vector<double> &dual,
                                  const ROL::Vector<double> &input_vector,
                                  const ROL::Vector<double> &des_var_sim,
                                  const ROL::Vector<double> &des_var_ctl,
                                  double &tol)
    {
        obj_->hessVec_11(output_vector, input_vector, des_var_sim, des_var_ctl, tol);
        output_vector.scale(getValue(dual));
    }

    /// Returns dual*dIdX2dX1*input_vector where dual should be a scalar, and input_vector has size X1.
    void applyAdjointHessian_12 ( ROL::Vector<double> &output_vector,
                                  const ROL::Vector<double> &dual,
                                  const ROL::Vector<double> &input_vector,
                                  const ROL::Vector<double> &des_var_sim,
                                  const ROL::Vector<double> &des_var_ctl,
                                  double &tol)
    {
        obj_->hessVec_21(output_vector, input_vector, des_var_sim, des_var_ctl, tol);
        output_vector.scale(getValue(dual));
    }

    /// Returns dual*dIdX1dX2*input_vector where dual should be a scalar, and input_vector has size X2.
    void applyAdjointHessian_21 ( ROL::Vector<double> &output_vector,
                                  const ROL::Vector<double> &dual,
                                  const ROL::Vector<double> &input_vector,
                                  const ROL::Vector<double> &des_var_sim,
                                  const ROL::Vector<double> &des_var_ctl,
                                  double &tol)
    {
        obj_->hessVec_12(output_vector, input_vector, des_var_sim, des_var_ctl, tol);
        output_vector.scale(getValue(dual));
    }

    /// Returns dual*dIdX2dX2*input_vector where dual should be a scalar, and input_vector has size X2.
    void applyAdjointHessian_22 ( ROL::Vector<double> &output_vector,
                                  const ROL::Vector<double> &dual,
                                  const ROL::Vector<double> &input_vector,
                                  const ROL::Vector<double> &des_var_sim,
                                  const ROL::Vector<double> &des_var_ctl,
                                  double &tol)
    {
        obj_->hessVec_22(output_vector, input_vector, des_var_sim, des_var_ctl, tol);
        output_vector.scale(getValue(dual));
    }

}; // ConstraintFromObjective_SimOpt

} // namespace PHiLiP
#endif
