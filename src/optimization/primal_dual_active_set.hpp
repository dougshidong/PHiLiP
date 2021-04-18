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

#ifndef PHILIP_PRIMALDUALACTIVESETSTEP_HPP
#define PHILIP_PRIMALDUALACTIVESETSTEP_HPP

#include "ROL_Step.hpp"
#include "ROL_Vector.hpp"
#include "ROL_KrylovFactory.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "ROL_Types.hpp"
#include "ROL_Secant.hpp"
#include "ROL_ParameterList.hpp"

/** @ingroup step_group
    \class ROL::PrimalDualActiveSetStep
    \brief Implements the computation of optimization steps 
           with the Newton primal-dual active set method.

    To describe primal-dual active set (PDAS), we consider the following 
    abstract setting.  Suppose \f$\mathcal{X}\f$ is a Hilbert space of 
    functions mapping \f$\Xi\f$ to \f$\mathbb{R}\f$.  For example, 
    \f$\Xi\subset\mathbb{R}^n\f$ and \f$\mathcal{X}=L^2(\Xi)\f$ or 
    \f$\Xi = \{1,\ldots,n\}\f$ and \f$\mathcal{X}=\mathbb{R}^n\f$. We 
    assume \f$ f:\mathcal{X}\to\mathbb{R}\f$ is twice-continuously Fr&eacute;chet 
    differentiable and \f$a,\,b\in\mathcal{X}\f$ with \f$a\le b\f$ almost 
    everywhere in \f$\Xi\f$.  Note that the PDAS algorithm will also work 
    with secant approximations of the Hessian. 

    Traditionally, PDAS is an algorithm for the minimizing quadratic objective 
    functions subject to bound constraints.  ROL implements a Newton PDAS which 
    extends PDAS to general bound-constrained nonlinear programs, i.e., 
    \f[
        \min_x \quad f(design_variables) \quad \text{s.t.} \quad a \le design_variables \le b.
    \f] 
    Given the \f$k\f$-th iterate \f$x_k\f$, the Newton PDAS algorithm computes 
    steps by applying PDAS to the quadratic subproblem 
    \f[
        \min_s \quad \langle \nabla^2 f(x_k)s + \nabla f(x_k),s \rangle_{\mathcal{X}}
        \quad \text{s.t.} \quad a \le x_k + s \le b.
    \f]
    For the \f$k\f$-th quadratic subproblem, PDAS builds an approximation of the 
    active set \f$\mathcal{A}_k\f$ using the dual variable \f$\lambda_k\f$ as 
    \f[
       \mathcal{A}^+_k = \{\,\xi\in\Xi\,:\,(\lambda_k + c(x_k-b))(\xi) > 0\,\}, \quad
       \mathcal{A}^-_k = \{\,\xi\in\Xi\,:\,(\lambda_k + c(x_k-a))(\xi) < 0\,\}, \quad\text{and}\quad
       \mathcal{A}_k = \mathcal{A}^-_k\cup\mathcal{A}^+_k.
    \f] 
    We define the inactive set \f$\mathcal{I}_k=\Xi\setminus\mathcal{A}_k\f$.
    The solution to the quadratic subproblem is then computed iteratively by solving 
    \f[
       \nabla^2 f(x_k) s_k + \dual_variables_{k+1} = -\nabla f(x_k), \quad
       x_k+s_k = a \;\text{on}\;\mathcal{A}^-_k,\quad x_k+s_k = b\;\text{on}\;\mathcal{A}^+_k,
       \quad\text{and}\quad
       \dual_variables_{k+1} = 0\;\text{on}\;\mathcal{I}_k
    \f]
    and updating the active and inactive sets. 
 
    One can rewrite this system by consolidating active and inactive parts, i.e., 
    \f[
       \begin{pmatrix}
           \nabla^2 f(x_k)_{\mathcal{A}_k,\mathcal{A}_k}  & \nabla^2 f(x_k)_{\mathcal{A}_k,\mathcal{I}_k} \\
           \nabla^2 f(x_k)_{\mathcal{I}_k,\mathcal{A}_k}  & \nabla^2 f(x_k)_{\mathcal{I}_k,\mathcal{I}_k} 
       \end{pmatrix}
       \begin{pmatrix}
         (s_k)_{\mathcal{A}_k} \\
         (s_k)_{\mathcal{I}_k}
       \end{pmatrix}
       +
       \begin{pmatrix}
         (\dual_variables_{k+1})_{\mathcal{A}_k} \\
         0
       \end{pmatrix}
       = - 
       \begin{pmatrix}
         \nabla f(x_k)_{\mathcal{A}_k}\\
         \nabla f(x_k)_{\mathcal{I}_k}
       \end{pmatrix}.
    \f]
    Here the subscripts \f$\mathcal{A}_k\f$ and \f$\mathcal{I}_k\f$ denote the active and inactive 
    components, respectively.  Moreover, the active components of \f$s_k\f$ are 
    \f$s_k(\xi) = a(\xi)-x_k(\xi)\f$ if \f$\xi\in\mathcal{A}^-_k\f$ and \f$s_k(\xi) = b(\xi)-x_k(\xi)\f$
    if \f$\xi\in\mathcal{A}^+_k\f$.  Since \f$(s_k)_{\mathcal{A}_k}\f$ is fixed, we only need to solve 
    for the inactive components of \f$s_k\f$ which we can do this using conjugate residuals (CR) (i.e., the 
    Hessian operator corresponding to the inactive indices may not be positive definite).  Once 
    \f$(s_k)_{\mathcal{I}_k}\f$ is computed, it is straight forward to update the dual variables.
*/

namespace PHiLiP {

template <class Real>
class PrimalDualActiveSetStep : public ROL::Step<Real>
{
private:

    ROL::Ptr<ROL::Krylov<Real> > krylov_;
  
    // Krylov Parameters
    int iter_Krylov_;  ///< CR iteration counter
    int flag_Krylov_;  ///< CR termination flag
    Real itol_;   ///< Inexact CR tolerance
  
    // PDAS Parameters
    int maxit_;      ///< Maximum number of PDAS iterations 
    int iter_PDAS_;       ///< PDAS iteration counter
    int flag_PDAS_;       ///< PDAS termination flag
    Real stol_;      ///< PDAS minimum step size stopping tolerance
    Real gtol_;      ///< PDAS gradient stopping tolerance
    Real scale_;     ///< Scale for dual variables in the active set, \f$c\f$, NOTE: papers usually scale primal variables, so this scale_ = 1/c of most papers from Hintermuller, Ito, Kunisch
    Real neps_;      ///< \f$\epsilon\f$-active set parameter 
    bool feasible_;  ///< Flag whether the current iterate is feasible or not
  
    // Dual Variable
    ROL::Ptr<ROL::Vector<Real> > dual_variables_;           ///< Container for dual variables
    ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;            ///< Container for primal plus dual variables
    ROL::Ptr<ROL::Vector<Real> > new_design_variables_;     ///< Container for new primal variables
    ROL::Ptr<ROL::Vector<Real> > search_temp_;              ///< Container for primal variable bounds
    ROL::Ptr<ROL::Vector<Real> > search_direction_active_set_;     ///< Container for step projected onto active set
    ROL::Ptr<ROL::Vector<Real> > desvar_tmp_;   ///< Container for temporary primal storage
    ROL::Ptr<ROL::Vector<Real> > quadratic_residual_;    ///< Container for optimality system residual for quadratic model
    ROL::Ptr<ROL::Vector<Real> > gradient_active_set_;     ///< Container for gradient projected onto active set
    ROL::Ptr<ROL::Vector<Real> > gradient_inactive_set_;     ///< Container for gradient projected onto active set
    ROL::Ptr<ROL::Vector<Real> > rhs_temp_;   ///< Container for temporary right hand side storage
    ROL::Ptr<ROL::Vector<Real> > gradient_tmp_;   ///< Container for temporary gradient storage
   
    // Secant Information
    ROL::ESecant esec_;                       ///< Enum for secant type
    ROL::Ptr<ROL::Secant<Real> > secant_; ///< Secant object
    bool useSecantPrecond_; 
    bool useSecantHessVec_;
  
    class InactiveHessian : public ROL::LinearOperator<Real> {
        private:
            const ROL::Ptr<ROL::Objective<Real> > obj_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bnd_;
            const ROL::Ptr<ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
            ROL::Ptr<ROL::Vector<Real> > v_;
            Real constraint_tolerance_;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            bool useSecant_;
        public:
          InactiveHessian(const ROL::Ptr<ROL::Objective<Real> > &objective,
                    const ROL::Ptr<ROL::BoundConstraint<Real> > &bnd,
                    const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                    const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                    const Real constraint_tolerance = 0,
                    const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                    const bool useSecant = false )
            : obj_(objective)
            , bnd_(bnd)
            , design_variables_(design_variables)
            , des_plus_dual_(des_plus_dual)
            , constraint_tolerance_(constraint_tolerance)
            , secant_(secant)
            , useSecant_(useSecant)
          {
              v_ = design_variables_->clone();
              if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
          }
          void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
          {
              v_->set(v);
              bnd_->pruneActive(*v_,*des_plus_dual_,constraint_tolerance_);
              if ( useSecant_ ) {
                  secant_->applyB(Hv,*v_);
              } else {
                  obj_->hessVec(Hv,*v_,*design_variables_,tol);
              }
              bnd_->pruneActive(Hv,*des_plus_dual_,constraint_tolerance_);
          }
    };
      
    class InactiveHessianPreconditioner : public ROL::LinearOperator<Real> {
        private:

            const ROL::Ptr<ROL::Objective<Real> > obj_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bnd_;
            const ROL::Ptr<ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
            ROL::Ptr<ROL::Vector<Real> > v_;
            Real constraint_tolerance_;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            bool useSecant_;

        public:
            InactiveHessianPreconditioner(const ROL::Ptr<ROL::Objective<Real> > &objective,
                      const ROL::Ptr<ROL::BoundConstraint<Real> > &bnd,
                      const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                      const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                      const Real constraint_tolerance = 0,
                      const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                      const bool useSecant = false )
              : obj_(objective)
              , bnd_(bnd)
              , design_variables_(design_variables)
              , des_plus_dual_(des_plus_dual)
              , constraint_tolerance_(constraint_tolerance)
              , secant_(secant)
              , useSecant_(useSecant)
            {
                v_ = design_variables_->dual().clone();
                if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
            }
            void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &/*tol*/ ) const
            {
                Hv.set(v.dual());
            }
            void applyInverse( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
            {
                v_->set(v);
                bnd_->pruneActive(*v_,*des_plus_dual_,constraint_tolerance_);
                if ( useSecant_ ) {
                    secant_->applyH(Hv,*v_);
                } else {
                    obj_->precond(Hv,*v_,*design_variables_,tol);
                }
                bnd_->pruneActive(Hv,*des_plus_dual_,constraint_tolerance_);
            }
    };
  
    /** \brief Compute the gradient-based criticality measure.
  
               The criticality measure is 
               \f$\|x_k - P_{[a,b]}(x_k-\nabla f(x_k))\|_{\mathcal{X}}\f$.
               Here, \f$P_{[a,b]}\f$ denotes the projection onto the
               bound constraints.
   
               @param[in]    design_variables     is the current iteration
               @param[in]    objective   is the objective function
               @param[in]    bound_constraints   are the bound constraints
               @param[in]    tol   is a tolerance for inexact evaluations of the objective function
    */ 
    Real computeCriticalityMeasure(
        ROL::Vector<Real> &design_variables,
        ROL::Objective<Real> &objective,
        ROL::BoundConstraint<Real> &bound_constraints,
        Real tol);
  
  public:
    /** \brief Constructor.
       
               @param[in]     parlist   is a parameter list containing relevent algorithmic information
               @param[in]     useSecant is a bool which determines whether or not the algorithm uses 
                                        a secant approximation of the Hessian
    */
    PrimalDualActiveSetStep( ROL::ParameterList &parlist );
  
    /** \brief Initialize step.  
  
               This includes projecting the initial guess onto the constraints, 
               computing the initial objective function value and gradient, 
               and initializing the dual variables.
  
               @param[in,out]    design_variables           is the initial guess 
               @param[in]        objective         is the objective function
               @param[in]        bound_constraints         are the bound constraints
               @param[in]        algo_state  is the current state of the algorithm
    */
    using ROL::Step<Real>::initialize;
    void initialize( ROL::Vector<Real> &design_variables, const ROL::Vector<Real> &search_direction, const ROL::Vector<Real> &g, 
                     ROL::Objective<Real> &objective, ROL::BoundConstraint<Real> &bound_constraints, 
                     ROL::AlgorithmState<Real> &algo_state );
  
    /** \brief Compute step.
  
               Given \f$x_k\f$, this function first builds the primal-dual active sets
               \f$\mathcal{A}_k^-\f$ and \f$\mathcal{A}_k^+\f$.  
               Next, it uses CR to compute the inactive components of the step by solving 
               \f[
                   \nabla^2 f(x_k)_{\mathcal{I}_k,\mathcal{I}_k}(s_k)_{\mathcal{I}_k}  = 
                       -\nabla f(x_k)_{\mathcal{I}_k}
                       -\nabla^2 f(x_k)_{\mathcal{I}_k,\mathcal{A}_k} (s_k)_{\mathcal{A}_k}.
               \f]
               Finally, it updates the active components of the dual variables as 
               \f[
                  \dual_variables_{k+1} = -\nabla f(x_k)_{\mathcal{A}_k} -(\nabla^2 f(x_k) s_k)_{\mathcal{A}_k}.
               \f]
  
               @param[out]       search_direction           is the step computed via PDAS
               @param[in]        design_variables           is the current iterate
               @param[in]        objective         is the objective function
               @param[in]        bound_constraints         are the bound constraints
               @param[in]        algo_state  is the current state of the algorithm
    */
    using ROL::Step<Real>::compute;
    void compute( ROL::Vector<Real> &search_direction,
                  const ROL::Vector<Real> &design_variables,
                  ROL::Objective<Real> &objective,
                  ROL::BoundConstraint<Real> &bound_constraints, 
                  ROL::AlgorithmState<Real> &algo_state );
  
    /** \brief Update step, if successful.
  
               This function returns \f$design_variables_{k+1} = x_k + s_k\f$.
               It also updates secant information if being used.
  
               @param[in]        design_variables           is the new iterate
               @param[out]       search_direction           is the step computed via PDAS
               @param[in]        objective         is the objective function
               @param[in]        bound_constraints         are the bound constraints
               @param[in]        algo_state  is the current state of the algorithm
    */
    using ROL::Step<Real>::update;
    void update( ROL::Vector<Real> &design_variables,
                 const ROL::Vector<Real> &search_direction,
                 ROL::Objective<Real> &objective,
                 ROL::BoundConstraint<Real> &bound_constraints,
                 ROL::AlgorithmState<Real> &algo_state );
  
    /** \brief Print iterate header.
  
               This function produces a string containing 
               header information.  
    */
    std::string printHeader( void ) const;
  
    /** \brief Print step name.
  
               This function produces a string containing 
               the algorithmic step information.  
    */
    std::string printName( void ) const;
  
    /** \brief Print iterate status.
      
               This function prints the iteration status.
  
               @param[in]        algo_state  is the current state of the algorithm
               @param[in]        printHeader if set to true will print the header at each iteration
    */
    virtual std::string print( ROL::AlgorithmState<Real> &algo_state, bool print_header = false ) const;

}; // class PrimalDualActiveSetStep

} // namespace PHiLiP

#endif

