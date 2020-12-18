#ifndef __PHILIP_CONSTRAINTFROMOBJECTIVE_SIMOPT_H__
#define __PHILIP_CONSTRAINTFROMOBJECTIVE_SIMOPT_H__
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_SingletonVector.hpp"

namespace PHiLiP {

/// Creates a constraint from an objective function and a offset value.
/** Same as ROL::ConstraintFromObjective except we do it for the SimOpt version.
 *  This will be especially useful for example when we have a target lift constraint
 *  such as CL - 0.375 = 0
 */
template<class Real> 
class ConstraintFromObjective_SimOpt : public ROL::Constraint_SimOpt<Real> {

    using V = ROL::Vector<Real>;

private:

    const ROL::Ptr<ROL::Objective_SimOpt<Real> > obj_;
    ROL::Ptr<V>                                  dualVector_1_;
    ROL::Ptr<V>                                  dualVector_2_;
    const Real                                   offset_;
    bool                                         isDual1Initialized_;
    bool                                         isDual2Initialized_;

    Real getValue( const V& x ) { 
        return dynamic_cast<const ROL::SingletonVector<Real>&>(x).getValue(); 
    }
 
    void setValue( V& x, Real val ) {
        dynamic_cast<ROL::SingletonVector<Real>&>(x).setValue(val);
    }

public:
    using ROL::Constraint_SimOpt<double>::value;
    using ROL::Constraint_SimOpt<double>::update;
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_1;
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_2;

    ConstraintFromObjective_SimOpt(
        const ROL::Ptr<ROL::Objective_SimOpt<Real> > &obj,
        const Real offset = 0 )
        : obj_(obj)
        , dualVector_1_(ROL::nullPtr)
        , dualVector_2_(ROL::nullPtr)
        , offset_(offset)
        , isDual1Initialized_(false)
        , isDual2Initialized_(false) {}

    const ROL::Ptr<ROL::Objective_SimOpt<Real> > getObjective(void) const { return obj_; }


    void setParameter( const std::vector<Real> &param ) override
    {
        obj_->setParameter(param);
        ROL::Constraint<Real>::setParameter(param);
    }


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
  
    void value( V& c, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        setValue(c, obj_->value(des_var_sim, des_var_ctl, tol) - offset_ ); 
    }

    void applyJacobian_1( V& jacobian_vector, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        if ( !isDual1Initialized_ ) {
          dualVector_1_ = des_var_sim.dual().clone();
          isDual1Initialized_ = true;
        }
        obj_->gradient_1(*dualVector_1_, des_var_sim, des_var_ctl, tol);
        setValue(jacobian_vector, v.dot(dualVector_1_->dual()));
    }
    void applyJacobian_2( V& jacobian_vector, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        if ( !isDual2Initialized_ ) {
          dualVector_2_ = des_var_ctl.dual().clone();
          isDual2Initialized_ = true;
        }
        obj_->gradient_2(*dualVector_2_, des_var_sim, des_var_ctl, tol);
        setValue(jacobian_vector, v.dot(dualVector_2_->dual()));
    }

    void applyAdjointJacobian_1( V& ajv, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        obj_->gradient_1(ajv, des_var_sim, des_var_ctl, tol);
        ajv.scale(getValue(v));
    }

    void applyAdjointJacobian_2( V& ajv, const V& v, const V& des_var_sim, const V& des_var_ctl, Real& tol ) override
    {
        obj_->gradient_2(ajv, des_var_sim, des_var_ctl, tol);
        ajv.scale(getValue(v));
    }

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
