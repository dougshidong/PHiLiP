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
// Original source file from Trilinos/packages/rol/src/step/ROL_PrimalDualActiveSetStep.hpp
// Modified for the PHiLiP project

#include "primal_dual_active_set.hpp"

namespace PHiLiP {

template <typename Real>
Real PrimalDualActiveSetStep<Real>::computeCriticalityMeasure(
    ROL::Vector<Real> &design_variables,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints,
    Real tol)
{
    Real one(1);
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    objective.gradient(*(step_state->gradientVec),design_variables,tol);
    desvar_tmp_->set(design_variables);
    desvar_tmp_->axpy(-one,(step_state->gradientVec)->dual());
    bound_constraints.project(*desvar_tmp_);
    desvar_tmp_->axpy(-one,design_variables);
    return desvar_tmp_->norm();
}
  

template <typename Real>
PrimalDualActiveSetStep<Real>::PrimalDualActiveSetStep( ROL::ParameterList &parlist ) 
    : ROL::Step<Real>::Step(), krylov_(ROL::nullPtr),
      iter_Krylov_(0), flag_Krylov_(0), itol_(0),
      maxit_(0), iter_PDAS_(0), flag_PDAS_(0), stol_(0), gtol_(0), scale_(0),
      neps_(-ROL::ROL_EPSILON<Real>()), feasible_(false),
      dual_variables_(ROL::nullPtr),
      des_plus_dual_(ROL::nullPtr),
      new_design_variables_(ROL::nullPtr),
      search_temp_(ROL::nullPtr),
      search_direction_active_set_(ROL::nullPtr),
      desvar_tmp_(ROL::nullPtr),
      quadratic_residual_(ROL::nullPtr),
      gradient_active_set_(ROL::nullPtr),
      gradient_inactive_set_(ROL::nullPtr),
      rhs_temp_(ROL::nullPtr),
      gradient_tmp_(ROL::nullPtr),
      esec_(ROL::SECANT_LBFGS), secant_(ROL::nullPtr), useSecantPrecond_(false),
      useSecantHessVec_(false)
{
    Real one(1), oem6(1.e-6), oem8(1.e-8);
    // ROL::Algorithmic parameters
    maxit_ = parlist.sublist("ROL::Step").sublist("Primal Dual Active Set").get("Iteration Limit",10);
    stol_ = parlist.sublist("ROL::Step").sublist("Primal Dual Active Set").get("Relative ROL::Step Tolerance",oem8);
    gtol_ = parlist.sublist("ROL::Step").sublist("Primal Dual Active Set").get("Relative Gradient Tolerance",oem6);
    scale_ = parlist.sublist("ROL::Step").sublist("Primal Dual Active Set").get("Dual Scaling", one);
    // Build secant object
    esec_ = ROL::StringToESecant(parlist.sublist("General").sublist("Secant").get("Type","Limited-Memory BFGS"));
    useSecantHessVec_ = parlist.sublist("General").sublist("Secant").get("Use as Hessian", false); 
    useSecantPrecond_ = parlist.sublist("General").sublist("Secant").get("Use as Preconditioner", false);
    if ( useSecantHessVec_ || useSecantPrecond_ ) {
      secant_ = ROL::SecantFactory<Real>(parlist);
    }
    // Build Krylov object
    krylov_ = ROL::KrylovFactory<Real>(parlist);
}

template<typename Real>
void PrimalDualActiveSetStep<Real>::initialize(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &search_direction,
    const ROL::Vector<Real> &g, 
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real zero(0), one(1);
    // Initialize state descent direction and gradient storage
    step_state->descentVec  = search_direction.clone();
    step_state->gradientVec = g.clone();
    step_state->searchSize  = zero;
    // Initialize additional storage
    des_plus_dual_ = design_variables.clone(); 
    new_design_variables_   = design_variables.clone();
    search_temp_ = design_variables.clone();
    search_direction_active_set_   = search_direction.clone(); 
    desvar_tmp_ = design_variables.clone(); 
    quadratic_residual_  = g.clone();
    gradient_active_set_   = g.clone(); 
    gradient_inactive_set_   = g.clone(); 
    rhs_temp_ = g.clone(); 
    gradient_tmp_ = g.clone(); 
    // Project design_variables onto constraint set
    bound_constraints.project(design_variables);
    // Update objective function, get value, and get gradient
    Real tol = std::sqrt(ROL::ROL_EPSILON<Real>());
    objective.update(design_variables,true,algo_state.iter);
    algo_state.value = objective.value(design_variables,tol);
    algo_state.nfval++;
    algo_state.gnorm = computeCriticalityMeasure(design_variables,objective,bound_constraints,tol);
    algo_state.ngrad++;
    // Initialize dual variable
    dual_variables_ = search_direction.clone(); 
    dual_variables_->set((step_state->gradientVec)->dual());
    dual_variables_->scale(-one);
}
  
template<typename Real>
void PrimalDualActiveSetStep<Real>::compute(
    ROL::Vector<Real> &search_direction,
    const ROL::Vector<Real> &design_variables,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real zero(0), one(1);
    search_direction.zero();
    new_design_variables_->set(design_variables);
    quadratic_residual_->set(*(step_state->gradientVec));
    // PDAS iterates through 3 steps.
    // 1. Estimate active set
    // 2. Use active set to determine search direction of the active constraints
    // 3. Solve KKT system for remaining inactive constraints
    for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {
        /********************************************************************/
        // Modify iterate vector to check active set
        /********************************************************************/
        des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
        des_plus_dual_->axpy(scale_,*(dual_variables_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply
        /********************************************************************/
        // Project design_variables onto primal dual feasible set
        // Using approximation of the active set, obtain the search direction since
        // we know that step will be constrained.
        /********************************************************************/
        search_direction_active_set_->zero();                                     // active_set_search_direction   = 0
    
        search_temp_->set(*bound_constraints.getUpperBound());                    // search_tmp = upper_bound
        search_temp_->axpy(-one,design_variables);                                // search_tmp = upper_bound - design_variables
        desvar_tmp_->set(*search_temp_);                                          // tmp        = upper_bound - design_variables
        bound_constraints.pruneUpperActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = (upper_bound - (upper_bound - design_variables + c*dual_variables)) < 0 ? 0 : upper_bound - design_variables
        search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(upper_bound - design_variables)

        search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(upper_bound - design_variables)
  
        search_temp_->set(*bound_constraints.getLowerBound());                    // search_tmp = lower_bound
        search_temp_->axpy(-one,design_variables);                                // search_tmp = lower_bound - design_variables
        desvar_tmp_->set(*search_temp_);                                          // tmp        = lower_bound - design_variables
        bound_constraints.pruneLowerActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = INACTIVE(lower_bound - design_variables)
        search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(lower_bound - design_variables)
        search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(lower_bound - design_variables)
        /********************************************************************/
        // Apply Hessian to active components of search_direction and remove inactive
        /********************************************************************/
        itol_ = std::sqrt(ROL::ROL_EPSILON<Real>());
        // INACTIVE(H)*active_set_search_direction = H*active_set_search_direction
        // INACTIVE(H)*active_set_search_direction = INACTIVE(H*active_set_search_direction)
        if ( useSecantHessVec_ && secant_ != ROL::nullPtr ) {
            secant_->applyB(*gradient_tmp_,*search_direction_active_set_);
        } else {
            objective.hessVec(*gradient_tmp_,*search_direction_active_set_,design_variables,itol_);
        }
        bound_constraints.pruneActive(*gradient_tmp_,*des_plus_dual_,neps_);
        /********************************************************************/
        // SEPARATE ACTIVE AND INACTIVE COMPONENTS OF THE GRADIENT
        /********************************************************************/
        // Inactive components
        gradient_inactive_set_->set(*(step_state->gradientVec));
        bound_constraints.pruneActive(*gradient_inactive_set_,*des_plus_dual_,neps_);
        // Active components
        gradient_active_set_->set(*(step_state->gradientVec));
        gradient_active_set_->axpy(-one,*gradient_inactive_set_);
        /********************************************************************/
        // SOLVE REDUCED NEWTON SYSTEM 
        /********************************************************************/

        // rhs_temp_ = -(INACTIVE(gradient) + INACTIVE(H*active_set_search_direction))
        rhs_temp_->set(*gradient_inactive_set_);
        rhs_temp_->plus(*gradient_tmp_);
        rhs_temp_->scale(-one);

        search_direction.zero();
        if ( rhs_temp_->norm() > zero ) {             
            // Initialize Hessian and preconditioner
            ROL::Ptr<ROL::Objective<Real> >       objective_ptr  = ROL::makePtrFromRef(objective);
            ROL::Ptr<ROL::BoundConstraint<Real> > constraint_ptr = ROL::makePtrFromRef(bound_constraints);
            ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<InactiveHessian>(objective_ptr, constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, secant_, useSecantHessVec_);
            ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<InactiveHessianPreconditioner>(objective_ptr, constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, secant_, useSecantPrecond_);
            krylov_->run(search_direction,*hessian,*rhs_temp_,*precond,iter_Krylov_,flag_Krylov_);
            bound_constraints.pruneActive(search_direction,*des_plus_dual_,neps_);        // search_direction <- inactive_search_direction
        }
        search_direction.plus(*search_direction_active_set_);                             // search_direction = inactive_search_direction + active_set_search_direction
        /********************************************************************/
        // UPDATE MULTIPLIER 
        /********************************************************************/
        if ( useSecantHessVec_ && secant_ != ROL::nullPtr ) {
            secant_->applyB(*rhs_temp_,search_direction);
        } else {
            objective.hessVec(*rhs_temp_,search_direction,design_variables,itol_);
        }
        gradient_tmp_->set(*rhs_temp_);
        bound_constraints.pruneActive(*gradient_tmp_,*des_plus_dual_,neps_);

        // dual^{k+1} = - ( ACTIVE(H * search_direction) + gradient_active_set_ )
        dual_variables_->set(*rhs_temp_);
        dual_variables_->axpy(-one,*gradient_tmp_);
        dual_variables_->plus(*gradient_active_set_);
        dual_variables_->scale(-one);
        /********************************************************************/
        // UPDATE STEP 
        /********************************************************************/
        new_design_variables_->set(design_variables);
        new_design_variables_->plus(search_direction);
        quadratic_residual_->set(*(step_state->gradientVec));
        quadratic_residual_->plus(*rhs_temp_);
        // Compute criticality measure  
        desvar_tmp_->set(*new_design_variables_);
        desvar_tmp_->axpy(-one,quadratic_residual_->dual());
        bound_constraints.project(*desvar_tmp_);
        desvar_tmp_->axpy(-one,*new_design_variables_);
        std::cout << "des_var_temp " << desvar_tmp_->norm() << std::endl;
        std::cout << "gtol gnorm " << gtol_*algo_state.gnorm << std::endl;
        std::cout << "rhs_temp_.norm() " << rhs_temp_->norm() << std::endl;
        if ( desvar_tmp_->norm() < gtol_*algo_state.gnorm ) {
            flag_PDAS_ = 0;
            break;
        }
        if ( search_direction.norm() < stol_*design_variables.norm() ) {
            flag_PDAS_ = 2;
            break;
        } 
    }
    if ( iter_PDAS_ == maxit_ ) {
        flag_PDAS_ = 1;
    } else {
        iter_PDAS_++;
    }
}
  
template<typename Real>
void PrimalDualActiveSetStep<Real>::update(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &search_direction,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints,
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    step_state->SPiter = (maxit_ > 1) ? iter_PDAS_ : iter_Krylov_;
    step_state->SPflag = (maxit_ > 1) ? flag_PDAS_ : flag_Krylov_;

    design_variables.plus(search_direction);
    feasible_ = bound_constraints.isFeasible(design_variables);
    algo_state.snorm = search_direction.norm();
    algo_state.iter++;
    Real tol = std::sqrt(ROL::ROL_EPSILON<Real>());
    objective.update(design_variables,true,algo_state.iter);
    algo_state.value = objective.value(design_variables,tol);
    algo_state.nfval++;
    
    if ( secant_ != ROL::nullPtr ) {
      gradient_tmp_->set(*(step_state->gradientVec));
    }
    algo_state.gnorm = computeCriticalityMeasure(design_variables,objective,bound_constraints,tol);
    algo_state.ngrad++;

    if ( secant_ != ROL::nullPtr ) {
      secant_->updateStorage(design_variables,*(step_state->gradientVec),*gradient_tmp_,search_direction,algo_state.snorm,algo_state.iter+1);
    }
    (algo_state.iterateVec)->set(design_variables);
}
  
template<typename Real>
std::string PrimalDualActiveSetStep<Real>::printHeader( void ) const 
{
    std::stringstream hist;
    hist << "  ";
    hist << std::setw(6) << std::left << "iter";
    hist << std::setw(15) << std::left << "value";
    hist << std::setw(15) << std::left << "gnorm";
    hist << std::setw(15) << std::left << "snorm";
    hist << std::setw(10) << std::left << "#fval";
    hist << std::setw(10) << std::left << "#grad";
    hist << std::setw(10) << std::left << "iterPDAS";
    hist << std::setw(10) << std::left << "flagPDAS";
    hist << std::setw(10) << std::left << "iterCR";
    hist << std::setw(10) << std::left << "flagCR";
    hist << std::setw(10) << std::left << "feasible";
    hist << "\n";
    return hist.str();
}
  
template<typename Real>
std::string PrimalDualActiveSetStep<Real>::printName( void ) const
{
    std::stringstream hist;
    hist << "\nPrimal Dual Active Set Newton's Method\n";
    return hist.str();
}
  
template<typename Real>
std::string PrimalDualActiveSetStep<Real>::print( ROL::AlgorithmState<Real> &algo_state, bool print_header ) const
{
    std::stringstream hist;
    hist << std::scientific << std::setprecision(6);
    if ( algo_state.iter == 0 ) hist << printName();
    if ( print_header ) hist << printHeader();
    if ( algo_state.iter == 0 ) {
        hist << "  ";
        hist << std::setw(6) << std::left << algo_state.iter;
        hist << std::setw(15) << std::left << algo_state.value;
        hist << std::setw(15) << std::left << algo_state.gnorm;
        hist << "\n";
    } else {
        hist << "  ";
        hist << std::setw(6) << std::left << algo_state.iter;
        hist << std::setw(15) << std::left << algo_state.value;
        hist << std::setw(15) << std::left << algo_state.gnorm;
        hist << std::setw(15) << std::left << algo_state.snorm;
        hist << std::setw(10) << std::left << algo_state.nfval;
        hist << std::setw(10) << std::left << algo_state.ngrad;
        hist << std::setw(10) << std::left << iter_PDAS_;
        hist << std::setw(10) << std::left << flag_PDAS_;
        hist << std::setw(10) << std::left << iter_Krylov_;
        hist << std::setw(10) << std::left << flag_Krylov_;
        if ( feasible_ ) {
            hist << std::setw(10) << std::left << "YES";
        } else {
            hist << std::setw(10) << std::left << "NO";
        }
        hist << "\n";
    }
    return hist.str();
}
  
template class PrimalDualActiveSetStep <double>;
} // namespace ROL

