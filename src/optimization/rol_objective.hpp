#ifndef __ROLOBJECTIVESIMOPT_H__
#define __ROLOBJECTIVESIMOPT_H__

#include "ROL_Objective_SimOpt.hpp"

#include "mesh/free_form_deformation.h"

#include "functional/functional.h"

// Constraint_SimOpt_from_Objective_SimOpt
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_SingletonVector.hpp"

namespace PHiLiP {

using ROL_Vector = ROL::Vector<double>;

/// Interface between the ROL::Objective_SimOpt PHiLiP::Functional.
/** Uses FFD to parametrize the geometry.
 *  An update on the simulation variables updates the DGBase object within the Functional
 *  and an update on the control variables updates the FreeFormDeformation object, which in
 *  turn, updates the DGBase.HighOrderGrid.volume_nodes.
 */
template <int dim, int nstate>
class ROLObjectiveSimOpt : public ROL::Objective_SimOpt<double> {
private:
    /// Functional to be evaluated
    Functional<dim,nstate,double> &functional;

    /// Free-form deformation used to parametrize the geometry.
    FreeFormDeformation<dim> ffd;

    /// List of FFD design variables and axes.
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;

    /// Design variables.
    dealii::LinearAlgebra::distributed::Vector<double> ffd_des_var;

    /// Design variables.
    dealii::LinearAlgebra::distributed::Vector<double> initial_ffd_des_var;

public:

    /// Stored mesh sensitivity evaluated at initialization.
    dealii::TrilinosWrappers::SparseMatrix dXvdXp;

    /// Constructor.
    ROLObjectiveSimOpt(
        Functional<dim,nstate,double> &_functional, 
        const FreeFormDeformation<dim> &_ffd,
        std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim,
        dealii::TrilinosWrappers::SparseMatrix *precomputed_dXvdXp = nullptr);
    ROLObjectiveSimOpt(
        Functional<dim,nstate,double> &_functional, 
        const FreeFormDeformation<dim> &_ffd,
        std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim,
        const dealii::TrilinosWrappers::SparseMatrix &_dXvdXp);
  
    using ROL::Objective_SimOpt<double>::value;
    using ROL::Objective_SimOpt<double>::update;
  
    /// Update the simulation and control variables.
    void update(
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        bool flag = true,
        int iter = -1) override;
  
    /// Returns the value of the Functional object.
    double value(
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;
  
    /// Returns the gradient w.\ r.\ t.\ the simulation variables of the Functional object.
    void gradient_1(
        ROL::Vector<double> &g,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;
  
    /// Returns the gradient w.\ r.\ t.\ the control variables of the Functional object.
    void gradient_2(
        ROL::Vector<double> &g,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;
  
    /// Applies the functional Hessian w.\ r.\ t.\ the simulation variables onto a vector.
    /** More specifically, 
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial w \partial w} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_11(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;
  
    /// Applies the functional Hessian w.\ r.\ t.\ the simulation and control variables onto a vector.
    /** More specifically, 
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial w \partial x} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_12(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;
  
    /// Applies the functional Hessian w.\ r.\ t.\ the control and simulation variables onto a vector.
    /** More specifically, 
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial x \partial w} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_21(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;
  
    /// Applies the functional Hessian w.\ r.\ t.\ the control variables onto a vector.
    /** More specifically, 
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial x \partial x} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_22(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ ) override;

}; // ROLObjectiveSimOpt

} // PHiLiP namespace

namespace ROL {

template<class Real> 
class Constraint_SimOpt_from_Objective_SimOpt : public Constraint_SimOpt<Real> {

    using V = ROL::Vector<Real>;

private:

    const ROL::Ptr<Objective_SimOpt<Real> > obj_;
    ROL::Ptr<V>                             dualVector_1_;
    ROL::Ptr<V>                             dualVector_2_;
    const Real                              offset_;
    bool                                    isDual1Initialized_;
    bool                                    isDual2Initialized_;

    Real getValue( const V& x ) { 
        return dynamic_cast<const SingletonVector<Real>&>(x).getValue(); 
    }
 
    void setValue( V& x, Real val ) {
        dynamic_cast<SingletonVector<Real>&>(x).setValue(val);
    }

public:

    Constraint_SimOpt_from_Objective_SimOpt(
        const ROL::Ptr<Objective_SimOpt<Real> > &obj,
        const Real offset = 0 )
        : obj_(obj)
        , dualVector_1_(ROL::nullPtr)
        , dualVector_2_(ROL::nullPtr)
        , offset_(offset)
        , isDual1Initialized_(false)
        , isDual2Initialized_(false) {}

    const ROL::Ptr<Objective_SimOpt<Real> > getObjective(void) const { return obj_; }


    void setParameter( const std::vector<Real> &param ) override
    {
        obj_->setParameter(param);
        Constraint<Real>::setParameter(param);
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

}; // Constraint_SimOpt_from_Objective_SimOpt

} // namespace ROL


#endif
