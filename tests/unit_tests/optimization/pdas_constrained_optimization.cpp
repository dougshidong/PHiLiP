#include <boost/program_options.hpp>

#include <fenv.h> // catch nan
// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <typeinfo>

#include <deal.II/base/utilities.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include <cmath>
#include <iostream>
#include <sstream>

#include "ROL_OptimizationProblem.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"

#include "ROL_Objective.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_Constraint.hpp"


#include "ROL_ConstraintStatusTest.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "optimization/primal_dual_active_set.hpp"
#include "optimization/optimization_utils.hpp"

enum class ConstraintType { linear, quadratic };
enum class ObjectiveType { quadratic, polynomial, rosenbrock };

//const ConstraintType constraint_type
//    = ConstraintType::quadratic;
//    //= ConstraintType::linear;
//const ObjectiveType objective_type
//    //= ObjectiveType::quadratic;
//    //= ObjectiveType::polynomial;
//    = ObjectiveType::rosenbrock;

//const int ITERATION_LIMIT = 2000;
const int ITERATION_LIMIT = 500;
const int LINESEARCH_MAX_ITER = 20;
const int PDAS_MAX_ITER = 1;
const int INCLUDE_SLACK_CONSTRAINTS = true;
const int INCLUDE_DESIGN_LOWER_BOUND_CONSTRAINTS = true;
const int INCLUDE_DESIGN_UPPER_BOUND_CONSTRAINTS = true;

// This test is used to check that the dealii::LinearAlgebra::distributed::Vector<double>
// is working properly with ROL. This is done by performing an unconstrained optimization
// of the Rosenbrock function.
using serial_Vector = dealii::Vector<double>;
using distributed_Vector = dealii::LinearAlgebra::distributed::Vector<double>;
// Use ROL to minimize the objective function, f(x,y) = x^2 + y^2.

/// Cast const ROL vector into a Teuchos::RCP pointer.
template <class VectorType, class Real>
Teuchos::RCP<const VectorType>
get_rcp_to_VectorType(const ROL::Vector<Real> &x)
{
    using AdaptVector = dealii::Rol::VectorAdaptor<VectorType>;
    return (Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
}

/// Cast ROL vector into a const Teuchos::RCP pointer.
template <class VectorType, class Real>
Teuchos::RCP<VectorType>
get_rcp_to_VectorType(ROL::Vector<Real> &x)
{
    using AdaptVector = dealii::Rol::VectorAdaptor<VectorType>;
    return (Teuchos::dyn_cast<AdaptVector>(x)).getVector();
}

/// Linear constraint function
template <class VectorType, class Real>
class LinearConstraint : public ROL::Constraint<Real>
{
public:
    /// Return the Linear constraint value.
    void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &x, Real & /*tol*/) override
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        Real sum = 0.0;

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (const auto& i : local_range) {
            sum += (*xp)[i];
        }
        const Real sum_all = dealii::Utilities::MPI::sum(sum, MPI_COMM_WORLD);
        c.setScalar(sum_all);
    }

    using ROL::Constraint<Real>::applyAdjointHessian;
    using ROL::Constraint<Real>::applyAdjointJacobian;
    using ROL::Constraint<Real>::applyJacobian;
    void applyAdjointJacobian(ROL::Vector<Real> &ajv,
                              const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &x,
                              const ROL::Vector<Real> &dualv,
                              Real &tol) override
    { 
        (void) tol; (void) dualv; (void) x;
        ajv.setScalar(1.0);
        const auto &v_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(v);
        Real val = v_singleton.getValue();
        ajv.scale(val);
    }

    void applyJacobian(ROL::Vector<Real> &av,
                              const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &x,
                              Real &tol) override
    { 
        (void) tol;
        Real val = x.dot(v);
        auto ones = x.clone();
        ones->setScalar(1.0);
        val = ones->dot(v);
        auto &av_singleton = dynamic_cast<ROL::SingletonVector<Real>&>(av);
        av_singleton.setValue(val);
    }

    void applyAdjointHessian(ROL::Vector<Real> &huv,
                             const ROL::Vector<Real> &u,
                             const ROL::Vector<Real> &v,
                             const ROL::Vector<Real> &x,
                             Real &tol ) override
    {
        (void) u; (void) v; (void) x; (void) tol;
        huv.zero();
    }

};

/// Quadratic constraint function
template <class VectorType, class Real>
class QuadraticConstraint : public ROL::Constraint<Real>
{
public:
    /// Return the Quadratic constraint value.
    void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &x, Real & /*tol*/) override
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);

        Real sqrsum = 0.0;

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            const Real &x0 = (*xp)[i];
            sqrsum += x0*x0;
        }
        const Real sqrsum_all = dealii::Utilities::MPI::sum(sqrsum, MPI_COMM_WORLD);
        c.setScalar(sqrsum_all);
    }

    using ROL::Constraint<Real>::applyAdjointHessian;
    using ROL::Constraint<Real>::applyAdjointJacobian;
    using ROL::Constraint<Real>::applyJacobian;
    void applyAdjointJacobian(ROL::Vector<Real> &ajv,
                              const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &x,
                              const ROL::Vector<Real> &dualv,
                              Real &tol) override
    { 
        (void) tol; (void) dualv;
        ajv.set(x);
        ajv.scale(2.0);
        const auto &v_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(v);
        Real val = v_singleton.getValue();
        ajv.scale(val);
    }

    void applyJacobian(ROL::Vector<Real> &av,
                              const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &x,
                              Real &tol) override
    { 
        (void) tol;
        //(void) dualv;
        Real val = x.dot(v);
        val *= 2.0;
        auto &av_singleton = dynamic_cast<ROL::SingletonVector<Real>&>(av);
        av_singleton.setValue(val);
    }

    void applyAdjointHessian(ROL::Vector<Real> &huv,
                             const ROL::Vector<Real> &u,
                             const ROL::Vector<Real> &v,
                             const ROL::Vector<Real> &x,
                             Real &tol ) override
    {
        (void) x; (void) tol;
        const auto &u_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(u);
        Real val = u_singleton.getValue();
        huv.set(v);
        huv.scale(2.0);
        huv.scale(val);
    }

};

/// Quadratic constraint function
template <class VectorType, class Real>
class ZeroConstraint : public ROL::Constraint<Real>
{
public:
    /// Return the Quadratic constraint value.
    void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &x, Real & /*tol*/) override
    {
        c.scale(0.0);
    }

    using ROL::Constraint<Real>::applyAdjointHessian;
    using ROL::Constraint<Real>::applyAdjointJacobian;
    using ROL::Constraint<Real>::applyJacobian;
    void applyAdjointJacobian(ROL::Vector<Real> &ajv,
                              const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &x,
                              const ROL::Vector<Real> &dualv,
                              Real &tol) override
    { 
        (void) tol; (void) dualv; (void) v; (void) x;
        ajv.scale(0.0);
    }

    void applyJacobian(ROL::Vector<Real> &av,
                              const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &x,
                              Real &tol) override
    { 
        (void) tol; (void) v; (void)x;
        av.scale(0.0);
    }

    void applyAdjointHessian(ROL::Vector<Real> &huv,
                             const ROL::Vector<Real> &u,
                             const ROL::Vector<Real> &v,
                             const ROL::Vector<Real> &x,
                             Real &tol ) override
    {
        (void) x; (void) tol;
        (void) u; (void) v;
        huv.scale(0.0);
    }

};

/// Quadratic objective function
template <class VectorType, class Real>
class QuadraticObjective : public ROL::Objective<Real>
{
public:
    /// Return the quadratic objective function value.
    Real value(const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);

        Real local_quadratic = 0.0;
        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        double quadratic_term = 1.0;
        for (const auto& i: local_range) {
            const Real &x0 = (*xp)[i];
            local_quadratic += 0.5*quadratic_term*(i+1)*(x0-1.0)*(x0-1.0);
        }
        const Real quad = dealii::Utilities::MPI::sum(local_quadratic, MPI_COMM_WORLD);
        return quad;

        return 0.0;
    }

    /// Return the quadratic objective gradient.
    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        Teuchos::RCP<VectorType>       gp = get_rcp_to_VectorType<VectorType,Real>(g);

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        double quadratic_term = 1.0;
        for (const auto& i: local_range) {
            const Real &x1 = (*xp)[i];
            Real dfdx = (i+1)*quadratic_term*(x1-1.0);
            (*gp)[i]  = dfdx;
        }
    }
    ///// Return the quadratic objective hessian vector product.
    void hessVec(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real & /*tol*/) override
    {
        (void) x;
        Teuchos::RCP<const VectorType> vp = get_rcp_to_VectorType<VectorType,Real>(v);
        Teuchos::RCP<VectorType>       hvp = get_rcp_to_VectorType<VectorType,Real>(hv);

        const dealii::IndexSet &local_range = (*vp).locally_owned_elements ();
        double quadratic_term = 1.0;
        for (const auto& i: local_range) {
            const Real &v1 = (*vp)[i];
            const Real dfdx = (i+1)*quadratic_term;
            (*hvp)[i]  = dfdx * v1;
        }
        return;
    }
};

/// Polynomial objective function
template <class VectorType, class Real>
class PolynomialObjective : public ROL::Objective<Real>
{
public:
    /// Return the polynomial objective function value.
    Real value(const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);

        Real local_quadratic = 0.0;
        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        double quadratic_term = 1.0;
        for (const auto& i: local_range) {
            const Real &x0 = (*xp)[i];
            double val = 1.0;
            for (int j = 0; j < 2; ++j) {
                val *= (x0-0.55);
            }
            for (int j = 0; j < 2; ++j) {
                val *= (x0-0.40);
            }
            local_quadratic += quadratic_term*val;
        }
        const Real quad = dealii::Utilities::MPI::sum(local_quadratic, MPI_COMM_WORLD);
        return quad;
    }

    /// Return the polynomial objective gradient.
    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        Teuchos::RCP<VectorType>       gp = get_rcp_to_VectorType<VectorType,Real>(g);

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        double quadratic_term = 1.0;
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const int i = *ip;
            const Real &x1 = (*xp)[i];
            Real dfdx = 0.0;
            double val = 1.0;
            val = 4 * (-0.1045 + 0.67125*x1 - 1.425*x1*x1 + x1*x1*x1);
            dfdx += val*quadratic_term;
            (*gp)[i]  = dfdx;
        }
    }
    /// Return the polynomial objective hessian vector product.
    void hessVec(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real & /*tol*/) override
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        Teuchos::RCP<const VectorType> vp = get_rcp_to_VectorType<VectorType,Real>(v);
        Teuchos::RCP<VectorType>       hvp = get_rcp_to_VectorType<VectorType,Real>(hv);

        const dealii::IndexSet &local_range = (*vp).locally_owned_elements ();
        double quadratic_term = 1.0;
        for (const auto& i: local_range) {
            const Real &x1 = (*xp)[i];
            const Real &v1 = (*vp)[i];
            const Real val = 4 * (0.67125 - 2*1.425*x1 + 3*x1*x1);
            const Real dfdx = val*quadratic_term;
            (*hvp)[i]  = dfdx * v1;
        }
        return;
    }
};


/// Rosensenbrock objective function
template <class VectorType, class Real>
class RosenbrockObjective : public ROL::Objective<Real>
{
public:
    /// Return the Rosenbrock objective function value.
    Real value(const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        xp->update_ghost_values();
        Real local_rosenbrock = 0.0;

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i == (*xp).size() - 1) continue;
            const Real &x0 = (*xp)[i];
            const Real &x1 = (*xp)[i+1];
            const Real term1 = std::pow(x1 - x0 * x0, 2);
            const Real term2 = std::pow(1.0 - x0, 2);
            local_rosenbrock += 100 * term1 + term2;
        }
        const Real rosenbrock = dealii::Utilities::MPI::sum(local_rosenbrock, MPI_COMM_WORLD);
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
            std::cout << std::setprecision(18);
            std::cout << "rosenbrock " << rosenbrock << std::endl;
        }
        return rosenbrock;
    }

    /// Return the Rosenbrock objective gradient.
    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        using FadType = Sacado::Fad::DFad<double>;


        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        Teuchos::RCP<VectorType>       gp = get_rcp_to_VectorType<VectorType,Real>(g);

        xp->update_ghost_values();

        (*gp) *= 0.0;
        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i==(*xp).size()-1) continue;
            const Real &x1 = (*xp)[i];
            const Real &x2 = (*xp)[i+1];
            // https://www.wolframalpha.com/input/?i=f%28a%2Cb%29+%3D+100*%28b-a*a%29%5E2+%2B+%281-a%29%5E2%2C+df%2Fda
            const Real drosenbrock_dx1 = 2.0*(200*x1*x1*x1 - 200*x1*x2 + x1 - 1.0);

            (*gp)[i]  = drosenbrock_dx1;
        }
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i==0) continue;
            const Real &x1 = (*xp)[i-1];
            const Real &x2 = (*xp)[i];
            const Real drosenbrock_dx2 = 200.0*(x2-x1*x1);
            (*gp)[i] += drosenbrock_dx2;
        }

        gp->update_ghost_values();

    }
    /// Return the quadratic objective hessian vector product.
    void hessVec(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real & /*tol*/) override
    //void hessVec2(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real & /*tol*/) 
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType<VectorType,Real>(x);
        Teuchos::RCP<const VectorType> vp = get_rcp_to_VectorType<VectorType,Real>(v);
        Teuchos::RCP<VectorType>       hvp = get_rcp_to_VectorType<VectorType,Real>(hv);

        xp->update_ghost_values();

		(*hvp) *= 0.0;
        const dealii::IndexSet &local_range = (*hvp).locally_owned_elements ();

        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
			
            if (i != 0 && i != (*xp).size()-1) {
                const Real &vl = (*vp)[i-1];
                const Real &vm = (*vp)[i];
                const Real &vr = (*vp)[i+1];

                const Real &xl = (*xp)[i-1];
                const Real &xm = (*xp)[i];
                const Real &xr = (*xp)[i+1];

                const Real diag = 1200 * xm*xm - 400 * xr + 2 + 200;
                const Real left_diag = -400 * xl;
                const Real right_diag = -400 * xm;

                std::cout << "Row: " << i << " Left: " << left_diag << " Diag: " << diag << " Right: " << right_diag << std::endl;
                (*hvp)[i] += left_diag * vl;
                (*hvp)[i] += diag * vm;
                (*hvp)[i] += right_diag * vr;
            } else if (i == 0) {
                const Real &vm = (*vp)[i];
                const Real &vr = (*vp)[i+1];

                const Real &xm = (*xp)[i];
                const Real &xr = (*xp)[i+1];

                const Real diag = 1200 * xm*xm - 400 * xr + 2 + 200;
                const Real right_diag = -400 * xm;

                std::cout << "Row: " << i << " Left: na Diag: " << diag << " Right: " << right_diag << std::endl;
                (*hvp)[i] += diag * vm;
                (*hvp)[i] += right_diag * vr;
            } else if (i == (*xp).size()-1) {
                const Real &vl = (*vp)[i-1];
                const Real &vm = (*vp)[i];

                const Real &xl = (*xp)[i-1];
                const Real &xm = (*xp)[i];

                const Real diag = 1200 * xm*xm + 2 + 200;
                const Real left_diag = -400 * xl;

                std::cout << "Row: " << i << " Left: " << left_diag << " Diag: " << diag << " Right: nothing" << std::endl;
                (*hvp)[i] += left_diag * vl;
                (*hvp)[i] += diag * vm;
            }
		}

        //for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
        //    const auto i = *ip;
        //    if (i==0) continue;
		//	
        //    const Real &v0 = (*vp)[i-1];
        //    const Real &v1 = (*vp)[i];

		//	const Real diag = 200;
		//	const Real lower_diag = -400 * (*xp)[i-1];
		//	const Real upper_diag = -400 * (*xp)[i-1];

        //    std::cout << "Row: " << i << " Lower: " << lower_diag << " Diag: " << diag << " Upper: " << upper_diag << std::endl;
        //    (*hvp)[i] += lower_diag * v0;
        //    (*hvp)[i] += diag * v1;
        //    (*hvp)[i-1] += upper_diag * v1;
		//}


        //for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
        //    const auto i = *ip;
        //    if (i==(*xp).size()-1) continue;
		//	
        //    const Real &v1 = (*vp)[i];
		//	const Real diag = 1200 * std::pow((*xp)[i], 2) - 400 * (*xp)[i+1] + 2;
        //    (*hvp)[i] += diag * v1;
		//}

        return;
    }
};

template <class VectorType, class Real = double>
ROL::Ptr<ROL::Objective<double>> getObjective(const ObjectiveType objective_type)
{
    switch(objective_type)
    {
        case(ObjectiveType::quadratic):
            return ROL::makePtr<QuadraticObjective<VectorType,Real>> ();
            break;
        case(ObjectiveType::polynomial):
            return ROL::makePtr<PolynomialObjective<VectorType,Real>> ();
            break;
        case(ObjectiveType::rosenbrock):
            return ROL::makePtr<RosenbrockObjective<VectorType,Real>> ();
            break;
        default:
            return ROL::makePtr<QuadraticObjective<VectorType,Real>> ();
    }
}

template <class VectorType, class Real = double>
std::vector<ROL::Ptr<ROL::Constraint<double>>> getInequalityConstraint(const ConstraintType constraint_type)
{

    ROL::Ptr<ROL::Constraint<double>> constraint;
    switch(constraint_type)
    {
        case(ConstraintType::linear):
            constraint = ROL::makePtr<LinearConstraint<VectorType,Real>> ();
            break;
        case(ConstraintType::quadratic):
            constraint = ROL::makePtr<QuadraticConstraint<VectorType,Real>> ();
            break;
        default:
            std::abort();
            break;
    }

    std::vector<ROL::Ptr<ROL::Constraint<double> > > cvec;
    cvec.push_back(constraint);

    return cvec;

}

std::vector<ROL::Ptr<ROL::Vector<double>>> getInequalityMultiplier() {
    std::vector<ROL::Ptr<ROL::Vector<double>>> emul;
    const ROL::Ptr<ROL::SingletonVector<double>> circle_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
    emul.push_back(circle_constraint_dual);
    return emul;
}

std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> getSlackBoundConstraint(ROL::Ptr<ROL::Vector<double>> design_variables, const ConstraintType constraint_type)
{
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> bcon;
    const ROL::Ptr<ROL::SingletonVector<double>> slack_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (-ROL::ROL_INF<double>());

    (void) constraint_type; (void) design_variables;
    const double multiple = 0.49;
    const double factor = (constraint_type == ConstraintType::linear) ? multiple : multiple*multiple;
    const unsigned int n = design_variables->dimension();
    const ROL::Ptr<ROL::SingletonVector<double>> slack_upper_bound = INCLUDE_SLACK_CONSTRAINTS ? 
                                                                     ROL::makePtr<ROL::SingletonVector<double>> (n*factor)
                                                                     : ROL::makePtr<ROL::SingletonVector<double>> (ROL::ROL_INF<double>());
    auto slack_bounds = ROL::makePtr<ROL::Bounds<double>> (slack_lower_bound, slack_upper_bound);
    bcon.push_back(slack_bounds);
    return bcon;
}
ROL::Ptr<ROL::BoundConstraint<double>> getBoundConstraint(ROL::Ptr<ROL::Vector<double>> design_variables) {
    struct setUpper : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setUpper() : zero_(0) {}
            double apply(const double &/*x*/) const {
                return INCLUDE_DESIGN_UPPER_BOUND_CONSTRAINTS ? 0.60 : ROL::ROL_INF<double>();
            }
    } setupper;
    struct setLower : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setLower() : zero_(0) {}
            double apply(const double &/*x*/) const {
                return INCLUDE_DESIGN_LOWER_BOUND_CONSTRAINTS ? 0.40 : -ROL::ROL_INF<double>();
            }
    } setlower;

    ROL::Ptr<ROL::Vector<double>> l = design_variables->clone();
    ROL::Ptr<ROL::Vector<double>> u = design_variables->clone();

    l->applyUnary(setlower);
    u->applyUnary(setupper);

    return ROL::makePtr<ROL::Bounds<double>>(l,u);
}


template <class VectorType>
int test(const unsigned int n_des_var, const ObjectiveType objective_type, const ConstraintType constraint_type, const bool use_bfgs)
{
    typedef double RealT;
  
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    int nmpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    if (mpi_rank == 0) std::cout << std::endl << std::endl;
    if (mpi_rank == 0) std::cout << "Optimization with " << n_des_var << " design variables using VectorType = " << typeid(VectorType).name() << std::endl;
  
    Teuchos::RCP<VectorType>   design_variables_rcp     = Teuchos::rcp(new VectorType);
  
    const dealii::IndexSet locally_owned_dofs = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n_des_var);
    if constexpr (std::is_same_v<VectorType, serial_Vector>) {
        design_variables_rcp->reinit(n_des_var);
    }
    if constexpr (std::is_same_v<VectorType, distributed_Vector>) {

        dealii::IndexSet ghost_dofs(n_des_var);
        for (unsigned int i=0; i<n_des_var; ++i) {
            ghost_dofs.add_index(i);
        }
        //for (auto ip = locally_owned_dofs.begin(); ip != locally_owned_dofs.end(); ++ip) {
        //    const auto index = *ip;

        //    const auto ghost_low = (index > 0) ? index-1 : 0;
        //    if (!locally_owned_dofs.is_element(ghost_low)) ghost_dofs.add_index(ghost_low);

        //    const auto ghost_high = (index < n_des_var-1) ? index+1 : n_des_var-1;
        //    if (!locally_owned_dofs.is_element(ghost_high)) ghost_dofs.add_index(ghost_high);
        //    
        //}
        design_variables_rcp->reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    }

  
    //for (auto xi = design_variables_rcp->begin(); xi != design_variables_rcp->end(); ++xi) {
    //    *xi = 0.5;
    //}
    //const dealii::IndexSet &local_range = (*design_variables_rcp).locally_owned_elements ();
    //for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
    //    const auto i = *ip;
    //    (*design_variables_rcp)[i] += i*0.01;
    //}

    design_variables_rcp->update_ghost_values();

    dealii::Rol::VectorAdaptor<VectorType> design_variables_rol(design_variables_rcp);
    auto design_variables_ptr = ROL::makePtrFromRef<dealii::Rol::VectorAdaptor<VectorType>>(design_variables_rol);
  
    // Set parameters.
    Teuchos::ParameterList parlist;
    parlist.sublist("Secant").set("Use as Preconditioner", false);
    parlist.sublist("Status Test").set("Gradient Tolerance", 1e-8);
    parlist.sublist("Status Test").set("Step Tolerance", 1e-14);
    parlist.sublist("Status Test").set("Iteration Limit", ITERATION_LIMIT);

    parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
    parlist.sublist("Step").sublist("Primal Dual Active Set").set("Relative Gradient Tolerance",1e-4);
    //parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory SR1");
    //parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory DFP");
    parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    parlist.sublist("General").sublist("Secant").set("Use as Hessian", use_bfgs);
    parlist.sublist("General").sublist("Secant").set("Maximum Storage",10);
    parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-12);
    parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-9);
    parlist.sublist("General").sublist("Krylov").set("Iteration Limit", 1000);
    parlist.sublist("General").sublist("Krylov").set("Use Initial Guess", true);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
    //const std::string line_search_curvature = "Null Curvature Condition";
    //const std::string line_search_curvature = "Strong Wolfe Conditions";
    const std::string line_search_curvature = "Wolfe Conditions";
    const std::string line_search_method = "Backtracking";
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


    ROL::Ptr<ROL::Objective<double>>                     objective              = getObjective<VectorType,RealT>(objective_type);
    ROL::Ptr<ROL::BoundConstraint<double>>               des_var_bound          = getBoundConstraint(design_variables_ptr);
    std::vector<ROL::Ptr<ROL::Constraint<double>>>       inequality_constraint  = getInequalityConstraint<VectorType,RealT>(constraint_type);
    std::vector<ROL::Ptr<ROL::Vector<double>>>           inequality_dual        = getInequalityMultiplier();
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>>  slack_bound            = getSlackBoundConstraint(design_variables_ptr, constraint_type);
    ROL::OptimizationProblem<RealT> opt_problem = ROL::OptimizationProblem<RealT> ( objective, design_variables_ptr, des_var_bound,
                                             inequality_constraint, inequality_dual, slack_bound);

  
    // Output stream
    ROL::nullstream bhs; // outputs nothing
    Teuchos::RCP<std::ostream> outStream;

    std::filebuf filebuffer;
    std::string otype_string, ctype_string;
    switch (objective_type) {
        case(ObjectiveType::quadratic) : otype_string = "quadratic"; break;
        case(ObjectiveType::rosenbrock) : otype_string = "rosenbrock"; break;
        case(ObjectiveType::polynomial) : otype_string = "polynomial"; break;
        default : std::abort(); break;
    }
    switch (constraint_type) {
        case(ConstraintType::quadratic) : ctype_string = "quadratic"; break;
        case(ConstraintType::linear) : ctype_string = "linear"; break;
        default : std::abort(); break;
    }

    std::string method = use_bfgs ? "_bfgs" : "_hessian";
    const std::string filename = "optimization_"
                                        +otype_string+"Objective_"
                                        +ctype_string+"Constraint"
                                        +"_nmpi"+std::to_string(nmpi)
                                        +method
                                        +"_ndes"+std::to_string(n_des_var)
                                        +".log";
    if (mpi_rank == 0) filebuffer.open (filename, std::ios::out);
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    ROL::Ptr<const ROL::AlgorithmState<RealT> > algo_state;

    auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
    auto status_test = ROL::makePtr<ROL::ConstraintStatusTest<double>>(parlist);
    const bool printHeader = true;
    const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
    auto gradient_vector = design_variables_ptr->dual().clone();

    auto x      = opt_problem.getSolutionVector();
    auto g      = x->dual().clone();
    auto l      = opt_problem.getMultiplierVector();
    auto c      = l->dual().clone();
    auto obj    = opt_problem.getObjective();
    auto con    = opt_problem.getConstraint();
    auto bnd    = opt_problem.getBoundConstraint();

    x->setScalar(0.5);
    l->setScalar(0.5);
    //for (auto &constraint_dual : inequality_dual) {
    //    constraint_dual->zero();
    //}
    algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
    algo_state = algorithm->getState();

    std::stringstream hist;
    hist << "Optimization Terminated with Status: ";
    hist << ROL::EExitStatusToString(algo_state->statusFlag);
    hist << "\n";
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if( comm_size == 1) {
      std::cout << hist.str() << std::endl;
    }

  
    Teuchos::RCP<const VectorType> xg = design_variables_rol.getVector();
    *outStream << "The solution to minimization problem is: ";
    std::cout << std::flush;

    if (mpi_rank == 0) filebuffer.close();

    auto write_out_vector = [&filename, &filebuffer](Teuchos::RCP<const VectorType> vec) {
        Teuchos::RCP<std::ostream> outStream;
        std::streamsize ss = std::cout.precision();
        std::cout.precision(15);
        for (unsigned int i=0; i<vec->size(); ++i) {

            if (vec->locally_owned_elements().is_element(i)) {
                filebuffer.open(filename, std::ios::app);
                std::ostream ostr2(&filebuffer);
                outStream = ROL::makePtrFromRef(ostr2);
                (*outStream).precision(15);
                std::cout << (*vec)[i] << " " << std::flush;
                *outStream << (*vec)[i] << " " << std::flush;

                filebuffer.close();
            }
            (void) MPI_Barrier(MPI_COMM_WORLD);
        }
        std::cout.precision(ss);
    };
    write_out_vector(xg);

    if (mpi_rank == 0) std::cout << "Design: " << std::endl;
    for (int i = 0; i < x->dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        const std::optional<double> value = PHiLiP::get_value(i, *x);
        if (value) {
            std::cout << *value << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) std::cout << std::endl;
    if (mpi_rank == 0) std::cout << std::endl;

    if (mpi_rank == 0) std::cout << "Constraint: " << std::endl;
    for (int i = 0; i < c->dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        const std::optional<double> value = PHiLiP::get_value(i, *c);
        if (value) {
            std::cout << *value << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) std::cout << std::endl;
    if (mpi_rank == 0) std::cout << std::endl;


    return algo_state->statusFlag;
}

int main (int argc, char * argv[])
{
    namespace po = boost::program_options;
    po::options_description description("Usage:");

    bool use_bfgs = false;
    description.add_options()
    ("help,h", "Solves a constrained optimization problem with the Primal Dual Active Set")
    ("objective_type,o", po::value<int>()->default_value(0),
     "Objective Type \n 0 = Quadratic \n 1 = Polynomial \n 2 = Rosenbrock")
    ("constraint_type,c", po::value<int>()->default_value(0),
     "Constraint Type \n 0 = Linear \n 1 = Quadratic")
    ("use_bfgs,u", po::bool_switch(&use_bfgs),
     "If --use_bfgs, use BFGS approximation, otherwise, use full exact Hessian.");
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
    po::notify(vm);
    if(vm.count("help")) {
        std::cout << description;
        std::abort();
    }

    const ObjectiveType objective_type  = static_cast<ObjectiveType>(vm["objective_type"].as<int>());
    const ConstraintType constraint_type = static_cast<ConstraintType>(vm["constraint_type"].as<int>());


    feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = false;
    try {
         test_error += test<distributed_Vector>(2, objective_type, constraint_type, use_bfgs);
         test_error += test<distributed_Vector>(3, objective_type, constraint_type, use_bfgs);
         test_error += test<distributed_Vector>(4, objective_type, constraint_type, use_bfgs);
         test_error += test<distributed_Vector>(5, objective_type, constraint_type, use_bfgs);
         test_error += test<distributed_Vector>(6, objective_type, constraint_type, use_bfgs);
         test_error += test<distributed_Vector>(7, objective_type, constraint_type, use_bfgs);
    }
    catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }

    return test_error;
}

