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


#ifndef ROL_REDUCED_OBJECTIVE_SIMOPT_FAILSAFE_H
#define ROL_REDUCED_OBJECTIVE_SIMOPT_FAILSAFE_H

#include "ROL_Reduced_Objective_SimOpt.hpp"

namespace ROL {

template <class Real>
class Reduced_Objective_SimOpt_FailSafe : public Reduced_Objective_SimOpt<Real> {

private:
  static constexpr double default_solver_tol = 1e-11;
  const Real solver_tol_;
public:

  /** \brief Constructor.

      @param[in] obj          is a pointer to a SimOpt objective function.
      @param[in] con          is a pointer to a SimOpt equality constraint.
      @param[in] state        is a pointer to a state space vector, \f$\mathcal{U}\f$.
      @param[in] control      is a pointer to a optimization space vector, \f$\mathcal{Z}\f$.
      @param[in] adjoint      is a pointer to a dual constraint space vector, \f$\mathcal{C}^*\f$.
      @param[in] storage      is a flag whether or not to store computed states and adjoints.
      @param[in] useFDhessVec is a flag whether or not to use a finite-difference Hessian approximation.
      @param[in] solver_tol   is the tolerance required of the solve_state_equation, otherwise return a large value
  */
  Reduced_Objective_SimOpt_FailSafe(
      const ROL::Ptr<Objective_SimOpt<Real> > &obj, 
      const ROL::Ptr<Constraint_SimOpt<Real> > &con, 
      const ROL::Ptr<Vector<Real> > &state, 
      const ROL::Ptr<Vector<Real> > &control, 
      const ROL::Ptr<Vector<Real> > &adjoint,
      const bool storage = true,
      const bool useFDhessVec = false,
      const Real solver_tol = default_solver_tol)
      : Reduced_Objective_SimOpt<Real>(obj, con, state, control, adjoint, storage, useFDhessVec), solver_tol_(solver_tol)
      {}

  /** \brief Secondary, general constructor for use with dual optimization vector spaces
             where the user does not define the dual() method.

      @param[in] obj          is a pointer to a SimOpt objective function.
      @param[in] con          is a pointer to a SimOpt equality constraint.
      @param[in] state        is a pointer to a state space vector, \f$\mathcal{U}\f$.
      @param[in] control      is a pointer to a optimization space vector, \f$\mathcal{Z}\f$.
      @param[in] adjoint      is a pointer to a dual constraint space vector, \f$\mathcal{C}^*\f$.
      @param[in] dualstate    is a pointer to a dual state space vector, \f$\mathcal{U}^*\f$.
      @param[in] dualadjoint  is a pointer to a constraint space vector, \f$\mathcal{C}\f$.
      @param[in] storage      is a flag whether or not to store computed states and adjoints.
      @param[in] useFDhessVec is a flag whether or not to use a finite-difference Hessian approximation.
      @param[in] solver_tol   is the tolerance required of the solve_state_equation, otherwise return a large value
  */
  Reduced_Objective_SimOpt_FailSafe(
      const ROL::Ptr<Objective_SimOpt<Real> > &obj,
      const ROL::Ptr<Constraint_SimOpt<Real> > &con,
      const ROL::Ptr<Vector<Real> > &state,
      const ROL::Ptr<Vector<Real> > &control, 
      const ROL::Ptr<Vector<Real> > &adjoint,
      const ROL::Ptr<Vector<Real> > &dualstate,
      const ROL::Ptr<Vector<Real> > &dualcontrol, 
      const ROL::Ptr<Vector<Real> > &dualadjoint,
      const bool storage = true,
      const bool useFDhessVec = false,
      const Real solver_tol = default_solver_tol)
    : Reduced_Objective_SimOpt<Real>(obj, con, state, control, adjoint, dualstate, dualcontrol, dualadjoint, storage, useFDhessVec),
      solver_tol_(solver_tol)
      {}

  /** \brief Constructor.

      @param[in] obj          is a pointer to a SimOpt objective function.
      @param[in] con          is a pointer to a SimOpt equality constraint.
      @param[in] stateStore   is a pointer to a SimController object.
      @param[in] state        is a pointer to a state space vector, \f$\mathcal{U}\f$.
      @param[in] control      is a pointer to a optimization space vector, \f$\mathcal{Z}\f$.
      @param[in] adjoint      is a pointer to a dual constraint space vector, \f$\mathcal{C}^*\f$.
      @param[in] storage      is a flag whether or not to store computed states and adjoints.
      @param[in] useFDhessVec is a flag whether or not to use a finite-difference Hessian approximation.
      @param[in] solver_tol   is the tolerance required of the solve_state_equation, otherwise return a large value
  */
  Reduced_Objective_SimOpt_FailSafe(
      const ROL::Ptr<Objective_SimOpt<Real> > &obj, 
      const ROL::Ptr<Constraint_SimOpt<Real> > &con, 
      const ROL::Ptr<SimController<Real> > &stateStore, 
      const ROL::Ptr<Vector<Real> > &state, 
      const ROL::Ptr<Vector<Real> > &control, 
      const ROL::Ptr<Vector<Real> > &adjoint,
      const bool storage = true,
      const bool useFDhessVec = false,
      const Real solver_tol = default_solver_tol)
    : Reduced_Objective_SimOpt<Real>(obj, con, stateStore, state, control, adjoint, storage, useFDhessVec), solver_tol_(solver_tol)
    {}

  /** \brief Secondary, general constructor for use with dual optimization vector spaces
             where the user does not define the dual() method.

      @param[in] obj          is a pointer to a SimOpt objective function.
      @param[in] con          is a pointer to a SimOpt equality constraint.
      @param[in] stateStore   is a pointer to a SimController object.
      @param[in] state        is a pointer to a state space vector, \f$\mathcal{U}\f$.
      @param[in] control      is a pointer to a optimization space vector, \f$\mathcal{Z}\f$.
      @param[in] adjoint      is a pointer to a dual constraint space vector, \f$\mathcal{C}^*\f$.
      @param[in] dualstate    is a pointer to a dual state space vector, \f$\mathcal{U}^*\f$.
      @param[in] dualadjoint  is a pointer to a constraint space vector, \f$\mathcal{C}\f$.
      @param[in] storage      is a flag whether or not to store computed states and adjoints.
      @param[in] useFDhessVec is a flag whether or not to use a finite-difference Hessian approximation.
      @param[in] solver_tol   is the tolerance required of the solve_state_equation, otherwise return a large value
  */
  Reduced_Objective_SimOpt_FailSafe(
      const ROL::Ptr<Objective_SimOpt<Real> > &obj,
      const ROL::Ptr<Constraint_SimOpt<Real> > &con,
      const ROL::Ptr<SimController<Real> > &stateStore, 
      const ROL::Ptr<Vector<Real> > &state,
      const ROL::Ptr<Vector<Real> > &control, 
      const ROL::Ptr<Vector<Real> > &adjoint,
      const ROL::Ptr<Vector<Real> > &dualstate,
      const ROL::Ptr<Vector<Real> > &dualcontrol, 
      const ROL::Ptr<Vector<Real> > &dualadjoint,
      const bool storage = true,
      const bool useFDhessVec = false,
      const Real solver_tol = default_solver_tol)
    : Reduced_Objective_SimOpt<Real>(obj, con, stateStore, state, control, adjoint, dualstate, dualcontrol, dualadjoint, storage, useFDhessVec),
      solver_tol_(solver_tol)
      {}

  /** \brief Given \f$z\in\mathcal{Z}\f$, evaluate the objective function 
             \f$\widehat{J}(z) = J(u(z),z)\f$ where 
             \f$u=u(z)\in\mathcal{U}\f$ solves \f$e(u,z) = 0\f$.
  */
  Real value( const Vector<Real> &z, Real &tol ) override {
    Real value = Reduced_Objective_SimOpt<Real>::value(z, tol);
    if (tol > solver_tol_) {
        std::cout << "NON-CONVERGED FLOW" << __PRETTY_FUNCTION__ << std::endl;
        return 1e99;
    }
    return value;
  }

  /** \brief Given \f$z\in\mathcal{Z}\f$, evaluate the gradient of the objective function 
             \f$\nabla\widehat{J}(z) = J_z(z) + c_z(u,z)^*\lambda\f$ where 
             \f$\lambda=\lambda(u,z)\in\mathcal{C}^*\f$ solves 
             \f$e_u(u,z)^*\lambda+J_u(u,z) = 0\f$.
  */
  void gradient( Vector<Real> &g, const Vector<Real> &z, Real &tol ) override {
    ROL::Reduced_Objective_SimOpt<Real>::gradient(g, z, tol);
    if (tol > solver_tol_) {
        std::cout << "NON-CONVERGED FLOW" << __PRETTY_FUNCTION__ << std::endl;
        g.setScalar(1e150);
        return;
    }
  }

}; // class Reduced_Objective_SimOpt_FailSafe

} // namespace ROL

#endif
