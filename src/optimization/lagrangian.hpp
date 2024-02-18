// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

// Modified by Doug Shi-Dong
// Original source file from Trilinos/packages/rol/src/function/objective/ROL_AugmentedLagrangian.hpp
// Modified for the PHiLiP project

#ifndef ROL_LAGRANGIAN_H
#define ROL_LAGRANGIAN_H

#include "ROL_Objective.hpp"
#include "ROL_Constraint.hpp"
#include "ROL_QuadraticPenalty.hpp"
#include "ROL_Vector.hpp"
#include "ROL_Types.hpp"
#include "ROL_ParameterList.hpp"
#include "ROL_Ptr.hpp"
#include <iostream>

/** @ingroup func_group
    \class PHiLiP::Lagrangian
    \brief Provides the interface to evaluate the Lagrangian.  
    
    Given a function
    \f$f:\mathcal{X}\to\mathbb{R}\f$ and an equality constraint
    \f$c:\mathcal{X}\to\mathcal{C}\f$, the Lagrangian functional is
    \f[
       L_A(x,\lambda,\mu) = f(x) + \langle \lambda, c(x)\rangle_{\mathcal{C}^*,\mathcal{C}}
    \f]
    where \f$\lambda\in\mathcal{C}^*\f$ denotes the Lagrange multiplier estimate.
*/


namespace PHiLiP {

template <class Real>
class Lagrangian : public Objective<Real> {
private:
  // Required for  Lagrangian definition
  const ROL::Ptr<Objective<Real> > obj_;
  const ROL::Ptr<Constraint<Real> > con_;
  ROL::Ptr<Vector<Real> > multiplier_;

  ROL::Ptr<Vector<Real> > conValue_;

  // Auxiliary storage
  ROL::Ptr<Vector<Real> > primalMultiplierVector_;
  ROL::Ptr<Vector<Real> > dualOptVector_;

  // Objective and constraint evaluations
  Real fval_;
  ROL::Ptr<Vector<Real> > gradient_;

  // Objective function scaling
  Real fscale_;
  // Constraint function scaling
  Real cscale_;

  // Evaluation counters
  int nfval_;
  int ngval_;
  int ncval_;

  // Flags to recompute quantities
  bool isValueComputed_;
  bool isObjGradientComputed_;
  bool isConstraintComputed_;
  bool isGradientComputed_;

  void evaluateConstraint(const Vector<Real> &x, Real &tol) {
    if ( !isConstraintComputed_ ) {
      // Evaluate constraint
      con_->value(*conValue_,x,tol);
      ncval_++;
      isConstraintComputed_ = true;
    }
  }

public:
  /** \brief Constructor.

      This creates a valid Lagrangian object.
      @param[in]          obj              is an objective function.
      @param[in]          con              is an equality constraint.
      @param[in]          mulitplier       is a Lagrange multiplier vector.
      @param[in]          penaltyParameter is the penalty parameter.
      @param[in]          optVec           is an optimization space vector.
      @param[in]          conVec           is a constraint space vector.
      @param[in]          parlist          is a parameter list.
  */
  Lagrangian(const ROL::Ptr<Objective<Real> > &obj,
             const ROL::Ptr<Constraint<Real> > &con,
             const Vector<Real> &multiplier,
             const Real penaltyParameter,
             const Vector<Real> &optVec,
             const Vector<Real> &conVec,
             ROL::ParameterList &parlist)
    : obj_(obj),
      fval_(0), fscale_(1),
      nfval_(0), ngval_(0),
      isValueComputed_(false),
      isObjGradientComputed_(false)
      isGradientComputed_(false)
      isConstraintComputed_(false)
  {

    gradient_      = optVec.dual().clone();
    dualOptVector_          = optVec.dual().clone();
    primalConVector_        = conVec.clone();
    conValue_               = conVec.clone();
    multiplier_             = multiplier.clone();
    primalMultiplierVector_ = multiplier.clone();

    ROL::ParameterList& sublist = parlist.sublist("Step").sublist(" Lagrangian");
    int HessianApprox = sublist.get("Level of Hessian Approximation",  0);

    pen_ = ROL::makePtr<QuadraticPenalty<Real>>(con,multiplier,penaltyParameter,optVec,conVec,scaleLagrangian_,HessianApprox);
  }

  /** \brief Null constructor.

      This constructor is only used for inheritance and does not create a
      valid Lagrangian object.  Do not use.
  */
  Lagrangian()
   : obj_(ROL::nullPtr),
     pen_(ROL::nullPtr),
     dualOptVector_(ROL::nullPtr),
     fval_(0),
     gradient_(ROL::nullPtr),
     fscale_(1), cscale_(1),
     nfval_(0), ngval_(0),
     isValueComputed_(false),
     isObjGradientComputed_(false)
     isGradientComputed_(false)
     isConstraintComputed_(false)
  {}

  virtual void update( const Vector<Real> &x, bool flag = true, int iter = -1 ) {
    obj_->update(x,flag,iter);
    pen_->update(x,flag,iter);

    isValueComputed_ = (flag ? false : isValueComputed_);
    isObjGradientComputed_ = (flag ? false : isObjGradientComputed_);
    isConstraintComputed_ = ( flag ? false : isConstraintComputed_ );
  }

  virtual Real value( const Vector<Real> &x, Real &tol ) {
    // Compute objective function value
    if ( !isValueComputed_ ) {
      fval_ = obj_->value(x,tol); nfval_++;
      isValueComputed_ = true;

      // Evaluate constraint
      evaluateConstraint(x,tol);
    }

    // Apply Lagrange multiplier to constraint
    Real cval = multiplier_->dot(conValue_->dual());

    return fscale_ * fval_ + cscale_ * cval;
  }

  virtual void gradient( Vector<Real> &g, const Vector<Real> &x, Real &tol ) {
    // Compute objective function gradient
    if ( !isObjGradientComputed_ ) {
      obj_->gradient(*gradient_,x,tol); ngval_++;
      isObjGradientComputed_ = true;
    }
    g.set(*gradient_);
    g.scale(fscale_);

    // Evaluate constraint
    evaluateConstraint(x,tol);
    // Compute gradient of Augmented Lagrangian
    primalMultiplierVector_->set(*multiplier_);
    primalMultiplierVector_->scale(cscale_);
    con_->applyAdjointJacobian(g,*primalMultiplierVector_,x,tol);

    // Compute gradient of  Lagrangian
    g.plus(*dualOptVector_);
  }

  virtual void hessVec( Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &x, Real &tol ) {
    // Apply objective Hessian to a vector
    obj_->hessVec(hv,v,x,tol);
    hv.scale(fscale_);
    // Apply penalty Hessian to a vector
    pen_->hessVec(*dualOptVector_,v,x,tol);
    // Build hessVec of  Lagrangian
    hv.plus(*dualOptVector_);
  }

  // Return objective function value
  virtual Real getObjectiveValue(const Vector<Real> &x) {
    Real tol = std::sqrt(ROL_EPSILON<Real>());
    // Evaluate objective function value
    if ( !isValueComputed_ ) {
      fval_ = obj_->value(x,tol); nfval_++;
      isValueComputed_ = true;
    }
    return fval_;
  }

  const Ptr<const Vector<Real>> getObjectiveGradient(const Vector<Real> &x) {
    Real tol = std::sqrt(ROL_EPSILON<Real>());
    // Compute objective function gradient
    if ( !isObjGradientComputed_ ) {
      obj_->gradient(*gradient_,x,tol); ngval_++;
      isObjGradientComputed_ = true;
    }
    return gradient_;
  }

  // Return constraint value
  virtual void getConstraintVec(Vector<Real> &c, const Vector<Real> &x) {
    pen_->getConstraintVec(c,x);
  }

  // Return total number of constraint evaluations
  virtual int getNumberConstraintEvaluations(void) const {
    return ncval_;
  }

  // Return total number of objective evaluations
  virtual int getNumberFunctionEvaluations(void) const {
    return nfval_;
  }

  // Return total number of gradient evaluations
  virtual int getNumberGradientEvaluations(void) const {
    return ngval_;
  }

  void setScaling(const Real cscale = 1) {
    cscale_ = cscale;
  }

  // Reset with upated penalty parameter
  virtual void reset(const Vector<Real> &multiplier, const Real penaltyParameter) {
    nfval_ = 0;
    ngval_ = 0;
    ncval_ = 0;
    multiplier_->set(multiplier);
    pen_->reset(multiplier,penaltyParameter);
  }
}; // class Lagrangian

} // namespace ROL

#endif

