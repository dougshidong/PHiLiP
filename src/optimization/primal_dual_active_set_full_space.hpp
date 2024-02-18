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

#include <deal.II/base/conditional_ostream.h>

#include "ROL_Step.hpp"
#include "ROL_Vector.hpp"
#include "ROL_KrylovFactory.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "ROL_Types.hpp"
#include "ROL_Secant.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_ParameterList.hpp"

#include "optimization/flow_constraints.hpp"

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
       \nabla^2 f(x_k) s_k + \dual_inequality_{k+1} = -\nabla f(x_k), \quad
       x_k+s_k = a \;\text{on}\;\mathcal{A}^-_k,\quad x_k+s_k = b\;\text{on}\;\mathcal{A}^+_k,
       \quad\text{and}\quad
       \dual_inequality_{k+1} = 0\;\text{on}\;\mathcal{I}_k
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
         (\dual_inequality_{k+1})_{\mathcal{A}_k} \\
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
class FullSpacePrimalDualActiveSetStep : public ROL::Step<Real>
{
private:
    /// Parameter list.
    ROL::ParameterList parlist_;

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
    int kkt_linesearches_;  ///< Number of linesearches done within the KKT iteration.

  
    // Dual Variable
    ROL::Ptr<ROL::Vector<Real> > dual_equality_;           ///< Container for dual variables
    ROL::Ptr<ROL::Vector<Real> > dual_inequality_;           ///< Container for dual variables
    ROL::Ptr<ROL::Vector<Real> > old_dual_inequality_;           ///< Container for dual variables
    ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;            ///< Container for primal plus dual variables
    ROL::Ptr<ROL::Vector<Real> > new_design_variables_;     ///< Container for new dual equality variables
    ROL::Ptr<ROL::Vector<Real> > new_dual_equality_;     ///< Container for new dual equality variables
    ROL::Ptr<ROL::Vector<Real> > search_temp_;              ///< Container for primal variable bounds
    ROL::Ptr<ROL::Vector<Real> > search_direction_active_set_;     ///< Container for step projected onto active set
    ROL::Ptr<ROL::Vector<Real> > desvar_tmp_;   ///< Container for temporary primal storage
    ROL::Ptr<ROL::Vector<Real> > quadratic_residual_;    ///< Container for optimality system residual for quadratic model
    ROL::Ptr<ROL::Vector<Real> > gradient_active_set_;     ///< Container for gradient projected onto active set
    ROL::Ptr<ROL::Vector<Real> > gradient_inactive_set_;     ///< Container for gradient projected onto active set
    ROL::Ptr<ROL::Vector<Real> > gradient_tmp1_;   ///< Container for temporary gradient storage
    ROL::Ptr<ROL::Vector<Real> > gradient_tmp2_;   ///< Container for temporary gradient storage
   

    ROL::Ptr<ROL::Vector<Real> > search_direction_dual_;  ///< Container for dual variable search direction

    // Secant Information
    ROL::ESecant esec_;                       ///< Enum for secant type
    ROL::Ptr<ROL::Secant<Real> > secant_; ///< Secant object
    bool useSecantPrecond_; 
    bool useSecantHessVec_;

    class PDAS_KKT_System : public ROL::LinearOperator<Real> {
        private:
            const ROL::Ptr<ROL::Objective<Real> > objective_;
            const ROL::Ptr<PHiLiP::FlowConstraints<PHILIP_DIM> > flow_constraints_;
            const ROL::Ptr<ROL::Constraint<Real> > equality_constraints_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
            const ROL::Ptr<const ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<const ROL::Vector<Real> > dual_equality_;
            const ROL::Ptr<const ROL::Vector<Real> > des_plus_dual_;

            ROL::Ptr<ROL::Vector<Real> > temp_design_;
            ROL::Ptr<ROL::Vector<Real> > temp_dual_equality_;
            ROL::Ptr<ROL::Vector<Real> > temp_dual_inequality_;

            ROL::Ptr<ROL::Vector<Real> > v_;

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
                    const Real constraint_tolerance = 0,
                    const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                    const bool useSecant = false );
          void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const;
    };
    class Identity_Preconditioner : public ROL::LinearOperator<Real> {
        public:
          void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
          {
              (void) v;
              (void) tol;
              Hv.set(v);
          }
    };

  
    class InactiveHessian : public ROL::LinearOperator<Real> {
        private:
            const ROL::Ptr<ROL::Objective<Real> > objective_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
            const ROL::Ptr<ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
            ROL::Ptr<ROL::Vector<Real> > v_;
            Real bounded_constraint_tolerance_;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            bool useSecant_;
        public:
          InactiveHessian(const ROL::Ptr<ROL::Objective<Real> > &objective,
                    const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                    const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                    const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                    const Real constraint_tolerance = 0,
                    const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                    const bool useSecant = false )
            : objective_(objective)
            , bound_constraints_(bound_constraints)
            , design_variables_(design_variables)
            , des_plus_dual_(des_plus_dual)
            , bounded_constraint_tolerance_(constraint_tolerance)
            , secant_(secant)
            , useSecant_(useSecant)
          {
              v_ = design_variables_->clone();
              if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
          }
          void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
          {
              v_->set(v);
              bound_constraints_->pruneActive(*v_,*des_plus_dual_,bounded_constraint_tolerance_);
              if ( useSecant_ ) {
                  secant_->applyB(Hv,*v_);
              } else {
                  objective_->hessVec(Hv,*v_,*design_variables_,tol);
                  //Hv.axpy(10.0,*v_);
              }
              bound_constraints_->pruneActive(Hv,*des_plus_dual_,bounded_constraint_tolerance_);
          }
    };
      
    class InactiveHessianPreconditioner : public ROL::LinearOperator<Real> {
        private:

            const ROL::Ptr<ROL::Objective<Real> > objective_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
            const ROL::Ptr<ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
            ROL::Ptr<ROL::Vector<Real> > v_;
            Real bounded_constraint_tolerance_;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            bool useSecant_;

        public:
            InactiveHessianPreconditioner(const ROL::Ptr<ROL::Objective<Real> > &objective,
                      const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                      const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                      const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                      const Real constraint_tolerance = 0,
                      const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                      const bool useSecant = false )
              : objective_(objective)
              , bound_constraints_(bound_constraints)
              , design_variables_(design_variables)
              , des_plus_dual_(des_plus_dual)
              , bounded_constraint_tolerance_(constraint_tolerance)
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
                bound_constraints_->pruneActive(*v_,*des_plus_dual_,bounded_constraint_tolerance_);
                if ( useSecant_ ) {
                    secant_->applyH(Hv,*v_);
                } else {
                    objective_->precond(Hv,*v_,*design_variables_,tol);
                }
                bound_constraints_->pruneActive(Hv,*des_plus_dual_,bounded_constraint_tolerance_);
            }
    };


protected:
    /// Used to prune active or inactive constraints values or dual values using the same
    /// BoundConstraint_Partitioned class.
    static ROL::PartitionedVector<Real> augment_constraint_to_design_and_constraint(
        const ROL::Ptr<ROL::Vector<Real>> vector_of_design_size,
        const ROL::Ptr<ROL::Vector<Real>> vector_of_constraint_size)
    {
        std::vector<ROL::Ptr<ROL::Vector<Real>>> vec_of_vec { vector_of_design_size, vector_of_constraint_size };
        ROL::PartitionedVector<Real> partitioned_vector( vec_of_vec );

        partitioned_vector[0].set(*vector_of_design_size);
        partitioned_vector[1].set(*vector_of_constraint_size);

        return partitioned_vector;
    }
    static void prune_active_constraints(const ROL::Ptr<ROL::Vector<Real>> constraint_vector_to_prune,
                                  const ROL::Ptr<const ROL::Vector<Real>> ref_controlslacks_vector,
                                  const ROL::Ptr<ROL::BoundConstraint<Real>> bound_constraints,
                                  Real /*eps*/)
    {
        const ROL::PartitionedVector<Real> &ref_controlslacks_vector_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*ref_controlslacks_vector);
        ROL::Ptr<ROL::Vector<Real>> design_vector = ref_controlslacks_vector_partitioned.get(0)->clone();

        ROL::PartitionedVector<Real> dummydes_constraint = augment_constraint_to_design_and_constraint( design_vector, constraint_vector_to_prune );
        
        bound_constraints->pruneActive(dummydes_constraint, ref_controlslacks_vector_partitioned);

        constraint_vector_to_prune->set(dummydes_constraint[1]);
    }
    static void prune_inactive_constraints(const ROL::Ptr<ROL::Vector<Real>> constraint_vector_to_prune,
                                  const ROL::Ptr<const ROL::Vector<Real>> ref_controlslacks_vector,
                                  const ROL::Ptr<ROL::BoundConstraint<Real>> bound_constraints)
    {
        const ROL::PartitionedVector<Real> &ref_controlslacks_vector_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*ref_controlslacks_vector);
        ROL::Ptr<ROL::Vector<Real>> design_vector = ref_controlslacks_vector_partitioned.get(0)->clone();

        ROL::PartitionedVector<Real> dummydes_constraint = augment_constraint_to_design_and_constraint( design_vector, constraint_vector_to_prune );
        
        bound_constraints->pruneInactive(dummydes_constraint, ref_controlslacks_vector_partitioned);

        constraint_vector_to_prune->set(dummydes_constraint[1]);
    }
    static ROL::Ptr<ROL::Vector<Real>> getOpt( ROL::Vector<Real> &xs )
    {
        return dynamic_cast<ROL::PartitionedVector<Real>&>(xs).get(0);
    }
    static ROL::Ptr<const ROL::Vector<Real>> getOpt( const ROL::Vector<Real> &xs )
    {
        return dynamic_cast<const ROL::PartitionedVector<Real>&>(xs).get(0);
    }
private:

    class InactiveConstrainedHessian : public ROL::LinearOperator<Real> {
        private:
            const ROL::Ptr<ROL::Objective<Real> > objective_;
            const ROL::Ptr<ROL::Constraint<Real> > equality_constraints_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
            const ROL::Ptr<ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<ROL::Vector<Real> > dual_equality_;
            const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;

            ROL::Ptr<ROL::Vector<Real> > temp_des_;
            ROL::Ptr<ROL::Vector<Real> > temp_dual_;

            ROL::Ptr<ROL::Vector<Real> > v_;
            const ROL::Ptr<ROL::Vector<Real> > inactive_input_des_;
            const ROL::Ptr<ROL::Vector<Real> > active_input_dual_;

            Real bounded_constraint_tolerance_;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            bool useSecant_;
        public:
          InactiveConstrainedHessian(
                    const ROL::Ptr<ROL::Objective<Real> > &objective,
                    const ROL::Ptr<ROL::Constraint<Real> > &equality_constraints,
                    const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                    const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                    const ROL::Ptr<ROL::Vector<Real> > &dual_equality,
                    const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                    const Real constraint_tolerance = 0,
                    const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                    const bool useSecant = false )
            : objective_(objective)
            , equality_constraints_(equality_constraints)
            , bound_constraints_(bound_constraints)
            , design_variables_(design_variables)
            , dual_equality_(dual_equality)
            , des_plus_dual_(des_plus_dual)
            , inactive_input_des_(design_variables_->clone())
            , active_input_dual_(dual_equality_->clone())
            , bounded_constraint_tolerance_(constraint_tolerance)
            , secant_(secant)
            , useSecant_(useSecant)
          {

              temp_des_ = design_variables_->clone();
              temp_dual_ = dual_equality_->clone();
              if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
          }
          void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
          {

              ROL::PartitionedVector<Real> &output_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);
              const ROL::PartitionedVector<Real> &input_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);

              const ROL::Ptr< const ROL::Vector< Real > > input_des = input_partitioned.get(0);
              const ROL::Ptr< const ROL::Vector< Real > > input_dual = input_partitioned.get(1);
              const ROL::Ptr< ROL::Vector< Real > > output_des = output_partitioned.get(0);
              const ROL::Ptr< ROL::Vector< Real > > output_dual = output_partitioned.get(1);

              inactive_input_des_->set(*input_des);
              bound_constraints_->pruneActive(*inactive_input_des_,*des_plus_dual_,bounded_constraint_tolerance_);

              //const ROL::PartitionedVector<Real> &inactive_input_des_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(inactive_input_des_);
              //inactive_input_des_partitioned.get(0);

              Real one(1);
              if ( useSecant_ ) {
                  secant_->applyB(*output_des,*inactive_input_des_);
              } else {
                  // Hv1 = H11 * v1
                  objective_->hessVec(*output_des,*inactive_input_des_,*design_variables_,tol);
                  equality_constraints_->applyAdjointHessian(*temp_des_, *dual_equality_, *inactive_input_des_, *design_variables_, tol);
                  output_des->axpy(one,*temp_des_);
                  //*output_des.axpy(10.0,*inactive_input_des_);
              }
              // Hv1 += H12 * v2

              active_input_dual_->set(*input_dual);
              FullSpacePrimalDualActiveSetStep<Real>::prune_inactive_constraints( active_input_dual_, des_plus_dual_, bound_constraints_);
              equality_constraints_->applyAdjointJacobian(*temp_des_, *active_input_dual_, *design_variables_, tol);
              output_des->axpy(one,*temp_des_);

              bound_constraints_->pruneActive(*output_des,*des_plus_dual_,bounded_constraint_tolerance_);

              equality_constraints_->applyJacobian(*output_dual, *input_des, *design_variables_, tol);
          }
    };
      
    class InactiveConstrainedHessianPreconditioner : public ROL::LinearOperator<Real> {
        private:

            const ROL::Ptr<ROL::Objective<Real> > objective_;
            const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
            const ROL::Ptr<ROL::Vector<Real> > design_variables_;
            const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
            ROL::Ptr<ROL::Vector<Real> > v_;
            Real bounded_constraint_tolerance_;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            bool useSecant_;

        public:
            InactiveConstrainedHessianPreconditioner(const ROL::Ptr<ROL::Objective<Real> > &objective,
                      const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                      const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                      const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                      const Real constraint_tolerance = 0,
                      const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                      const bool useSecant = false )
              : objective_(objective)
              , bound_constraints_(bound_constraints)
              , design_variables_(design_variables)
              , des_plus_dual_(des_plus_dual)
              , bounded_constraint_tolerance_(constraint_tolerance)
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
                //v_->set(v);
                //bound_constraints_->pruneActive(*v_,*des_plus_dual_,bounded_constraint_tolerance_);
                //if ( useSecant_ ) {
                //    secant_->applyH(Hv,*v_);
                //} else {
                //    objective_->precond(Hv,*v_,*design_variables_,tol);
                //}
                //bound_constraints_->pruneActive(Hv,*des_plus_dual_,bounded_constraint_tolerance_);
                (void) tol;
                Hv.set(v.dual());
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
    FullSpacePrimalDualActiveSetStep( ROL::ParameterList &parlist );
  
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
    void initialize( ROL::Vector<Real> &design_variables,
                     const ROL::Vector<Real> &search_direction_vec_to_clone,
                     const ROL::Vector<Real> &gradient_vec_to_clone, 
                     ROL::Objective<Real> &objective,
                     ROL::BoundConstraint<Real> &bound_constraints, 
                     ROL::AlgorithmState<Real> &algo_state );

    void initialize(ROL::Vector<Real> &design_variables,
                    const ROL::Vector<Real> &gradient_vec_to_clone, 
                    ROL::Vector<Real> &dual_equality,
                    const ROL::Vector<Real> &constraint_vec_to_clone, 
                    ROL::Objective<Real> &objective,
                    ROL::Constraint<Real> &equality_constraints, 
                    ROL::BoundConstraint<Real> &bound_constraints, 
                    ROL::AlgorithmState<Real> &algo_state ) override;

  
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
                  \dual_inequality_{k+1} = -\nabla f(x_k)_{\mathcal{A}_k} -(\nabla^2 f(x_k) s_k)_{\mathcal{A}_k}.
               \f]
  
               @param[out]       search_direction_design           is the step computed via PDAS
               @param[in]        design_variables           is the current iterate
               @param[in]        objective         is the objective function
               @param[in]        bound_constraints         are the bound constraints
               @param[in]        algo_state  is the current state of the algorithm
    */
    using ROL::Step<Real>::compute;
    void compute( ROL::Vector<Real> &search_direction_design,
                  const ROL::Vector<Real> &design_variables,
                  ROL::Objective<Real> &objective,
                  ROL::BoundConstraint<Real> &bound_constraints, 
                  ROL::AlgorithmState<Real> &algo_state );

    void compute_PDAS_rhs(
        const ROL::Vector<Real> &old_design_variables,
        const ROL::Vector<Real> &new_design_variables,
        const ROL::Vector<Real> &new_dual_equality,
        const ROL::Vector<Real> &dual_inequality,
        const ROL::Vector<Real> &des_plus_dual,
        const ROL::Vector<Real> &old_objective_gradient,
        ROL::Objective<Real> &objective, 
        ROL::Constraint<Real> &equality_constraints, 
        ROL::BoundConstraint<Real> &bound_constraints, 
        ROL::PartitionedVector<Real> &rhs_partitioned);
    void compute2(
        ROL::Vector<Real> &search_direction_design,
        const ROL::Vector<Real> &design_variables,
        const ROL::Vector<Real> &dual_equality,
        ROL::Objective<Real> &objective,
        ROL::Constraint<Real> &equality_constraints, 
        ROL::BoundConstraint<Real> &bound_constraints, 
        ROL::AlgorithmState<Real> &algo_state );
    void compute(
        ROL::Vector<Real> &search_direction_design,
        const ROL::Vector<Real> &design_variables,
        const ROL::Vector<Real> &dual_equality,
        ROL::Objective<Real> &objective,
        ROL::Constraint<Real> &equality_constraints, 
        ROL::BoundConstraint<Real> &bound_constraints, 
        ROL::AlgorithmState<Real> &algo_state );
  
    /** \brief Update step, if successful.
  
               This function returns \f$design_variables_{k+1} = x_k + s_k\f$.
               It also updates secant information if being used.
  
               @param[in]        design_variables           is the new iterate
               @param[out]       search_direction_design           is the step computed via PDAS
               @param[in]        objective         is the objective function
               @param[in]        bound_constraints         are the bound constraints
               @param[in]        algo_state  is the current state of the algorithm
    */
    using ROL::Step<Real>::update;
    void update( ROL::Vector<Real> &design_variables,
                 const ROL::Vector<Real> &search_direction_design,
                 ROL::Objective<Real> &objective,
                 ROL::BoundConstraint<Real> &bound_constraints,
                 ROL::AlgorithmState<Real> &algo_state );

    void update(
        ROL::Vector<Real> &design_variables,
        ROL::Vector<Real> &dual_equality,
        const ROL::Vector<Real> &search_direction_design,
        ROL::Objective<Real> &objective,
        ROL::Constraint<Real> &equality_constraints,
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

private:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // class FullSpacePrimalDualActiveSetStep

} // namespace PHiLiP

#endif


