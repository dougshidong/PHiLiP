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

enum class ConstraintType { linear, quadratic };
enum class ObjectiveType { quadratic, polynomial, rosenbrock };

const ConstraintType constraint_type
    = ConstraintType::quadratic;
    //= ConstraintType::linear;
const ObjectiveType objective_type
    //= ObjectiveType::quadratic;
    //= ObjectiveType::polynomial;
    = ObjectiveType::rosenbrock;

const bool USE_BFGS = true;
const int LINESEARCH_MAX_ITER = 5;
const int PDAS_MAX_ITER = 7;

// This test is used to check that the dealii::LinearAlgebra::distributed::Vector<double>
// is working properly with ROL. This is done by performing an unconstrained optimization
// of the Rosenbrock function.
using serial_Vector = typename dealii::Vector<double>;
using distributed_Vector = typename dealii::LinearAlgebra::distributed::Vector<double>;
// Use ROL to minimize the objective function, f(x,y) = x^2 + y^2.

/// Rosensenbrock objective function
template <typename VectorType, class Real = double, typename AdaptVector = dealii::Rol::VectorAdaptor<VectorType>>
class RosenbrockConstraint : public ROL::Constraint<Real>
{
private:
    /// Cast const ROL vector into a Teuchos::RCP pointer.
    Teuchos::RCP<const VectorType>
    get_rcp_to_VectorType(const ROL::Vector<Real> &x)
    {
      return (Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
    }

    /// Cast ROL vector into a const Teuchos::RCP pointer.
    Teuchos::RCP<VectorType>
    get_rcp_to_VectorType(ROL::Vector<Real> &x)
    {
      return (Teuchos::dyn_cast<AdaptVector>(x)).getVector();
    }
public:
    /// Return the Rosenbrock objective function value.
    void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &x, Real & /*tol*/) override
    {
        Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
        // Rosenbrock function

        Real sqrsum = 0.0;

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            const Real &x0 = (*xp)[i];
            if (constraint_type == ConstraintType::quadratic) {
                sqrsum += x0*x0;
            } else if (constraint_type == ConstraintType::linear) {
                sqrsum += x0;
            }
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
        (void) tol;
        (void) dualv;
        if (constraint_type == ConstraintType::quadratic) {
            ajv.set(x);
            ajv.scale(2.0);
        } else if (constraint_type == ConstraintType::linear) {
            ajv.setScalar(1.0);
        }
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
        if (constraint_type == ConstraintType::quadratic) {
            val *= 2.0;
        } else if (constraint_type == ConstraintType::linear) {
            auto ones = x.clone();
            ones->setScalar(1.0);
            val = ones->dot(v);
        }
        auto &av_singleton = dynamic_cast<ROL::SingletonVector<Real>&>(av);
        av_singleton.setValue(val);
    }

    void applyAdjointHessian(ROL::Vector<Real> &huv,
                             const ROL::Vector<Real> &u,
                             const ROL::Vector<Real> &v,
                             const ROL::Vector<Real> &x,
                             Real &tol ) override
    {
        if (constraint_type == ConstraintType::quadratic) {
            const auto &u_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(u);
            Real val = u_singleton.getValue();
            huv.set(v);
            huv.scale(2.0);
            huv.scale(val);
            (void) x;
            (void) tol;
        } else if (constraint_type == ConstraintType::linear) {
            huv.zero();
        }
    }

    /// Return the Rosenbrock objective gradient.
    //void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    //{
    //  using FadType = Sacado::Fad::DFad<double>;


    //  Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
    //  Teuchos::RCP<VectorType>       gp = this->get_rcp_to_VectorType(g);

    //  (*gp) *= 0.0;
    //  const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
    //  for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
    //      const auto i = *ip;
    //      const Real &x1 = (*xp)[i];
    //      const Real dsqrsum_dx1 = 2.0*x1;
    //      (*gp)[i]  = dsqrsum_dx1;
    //  }
    //}
};


/// Rosensenbrock objective function
template <typename VectorType, class Real = double, typename AdaptVector = dealii::Rol::VectorAdaptor<VectorType>>
class RosenbrockObjective : public ROL::Objective<Real>
{
private:
    /// Cast const ROL vector into a Teuchos::RCP pointer.
    Teuchos::RCP<const VectorType>
    get_rcp_to_VectorType(const ROL::Vector<Real> &x)
    {
      return (Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
    }

    /// Cast ROL vector into a const Teuchos::RCP pointer.
    Teuchos::RCP<VectorType>
    get_rcp_to_VectorType(ROL::Vector<Real> &x)
    {
      return (Teuchos::dyn_cast<AdaptVector>(x)).getVector();
    }

public:
    /// Return the Rosenbrock objective function value.
    Real value(const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
        if (objective_type == ObjectiveType::quadratic) {
            Real local_quadratic = 0.0;
            const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
            double quadratic_term = 1.0;
            int powerr = 1; (void) powerr;
            for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
                const auto i = *ip;
                const Real &x0 = (*xp)[i];
                local_quadratic += 0.5*quadratic_term*(x0-1.0)*(x0-1.0);
                quadratic_term *= 0.85;
            }
            const Real quad = dealii::Utilities::MPI::sum(local_quadratic, MPI_COMM_WORLD);
            return quad;
        } else if (objective_type == ObjectiveType::polynomial) {
            Real local_quadratic = 0.0;
            const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
            double quadratic_term = 1.0;
            int powerr = 1; (void) powerr;
            for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
                const auto i = *ip;
                const Real &x0 = (*xp)[i];
                double val = 1.0;
                for (int i = 0; i <= powerr; ++i) {
                    val *= (x0-1.0);
                }
                powerr++;
                local_quadratic += quadratic_term*val;
                quadratic_term *= 0.85;
            }
            const Real quad = dealii::Utilities::MPI::sum(local_quadratic, MPI_COMM_WORLD);
            return quad;
        } else if (objective_type == ObjectiveType::rosenbrock) {

            // Rosenbrock function
            Real local_rosenbrock = 0.0;

            const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
            for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
                const auto i = *ip;
                if (i == (*xp).size() - 1) continue;
                const Real &x0 = (*xp)[i];
                const Real &x1 = (*xp)[i+1];
                local_rosenbrock += 100*(x1 - x0*x0)*(x1 - x0*x0) + (1.0-x0)*(1.0-x0);
            }
            const Real rosenbrock = dealii::Utilities::MPI::sum(local_rosenbrock, MPI_COMM_WORLD);
            return rosenbrock;
        }

        return 0.0;
    }

    /// Return the Rosenbrock objective gradient.
    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        using FadType = Sacado::Fad::DFad<double>;


        Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
        Teuchos::RCP<VectorType>       gp = this->get_rcp_to_VectorType(g);

        if (objective_type == ObjectiveType::quadratic) {
            const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
            double quadratic_term = 1.0;
            int powerr = 1; (void) powerr;
            for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
                const auto i = *ip;
                const Real &x1 = (*xp)[i];
                Real drosenbrock_dx1 = 0.0;
                drosenbrock_dx1 = quadratic_term*(x1-1.0);
                quadratic_term *= 0.85;
                (*gp)[i]  = drosenbrock_dx1;
            }
            return;
        } else if (objective_type == ObjectiveType::polynomial) {
            const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
            double quadratic_term = 1.0;
            int powerr = 1; (void) powerr;
            for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
                const auto i = *ip;
                const Real &x1 = (*xp)[i];
                Real drosenbrock_dx1 = 0.0;
                double val = 1.0;
                for (int i = 0; i <= powerr-1; ++i) {
                    val *= (x1-1.0);
                }
                drosenbrock_dx1 += val*powerr*quadratic_term;
                powerr++;
                quadratic_term *= 0.85;
                (*gp)[i]  = drosenbrock_dx1;
            }
            return;
        } else if (objective_type == ObjectiveType::rosenbrock) {

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
        }
    }
};

template <typename VectorType, class Real = double>
std::vector<ROL::Ptr<ROL::Constraint<double>>> getInequalityConstraint() {
    const ROL::Ptr<ROL::Constraint<double>> rosenbrock_constraint = ROL::makePtr<RosenbrockConstraint<VectorType,Real>> ();
    std::vector<ROL::Ptr<ROL::Constraint<double> > > cvec;
    cvec.push_back(rosenbrock_constraint);

    return cvec;

}

std::vector<ROL::Ptr<ROL::Vector<double>>> getInequalityMultiplier() {
    std::vector<ROL::Ptr<ROL::Vector<double>>> emul;
    const ROL::Ptr<ROL::SingletonVector<double>> circle_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
    emul.push_back(circle_constraint_dual);
    return emul;
}

std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> getSlackBoundConstraint(ROL::Ptr<ROL::Vector<double>> design_variables) 
{
    const unsigned int n = design_variables->dimension();
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> bcon;
    const ROL::Ptr<ROL::SingletonVector<double>> circle_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (-ROL::ROL_INF<double>());
    //const ROL::Ptr<ROL::SingletonVector<double>> circle_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (-1e-4);

    //const ROL::Ptr<ROL::SingletonVector<double>> circle_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (ROL::ROL_INF<double>());
    double multiple = 0.49;
    if (constraint_type == ConstraintType::quadratic) {
        multiple = multiple * multiple;
    }
    const ROL::Ptr<ROL::SingletonVector<double>> circle_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (n*multiple);
    //const ROL::Ptr<ROL::SingletonVector<double>> circle_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (0.9);
    auto circle_bounds = ROL::makePtr<ROL::Bounds<double>> (circle_lower_bound, circle_upper_bound);
    bcon.push_back(circle_bounds);
    return bcon;
}
ROL::Ptr<ROL::BoundConstraint<double>> getBoundConstraint(ROL::Ptr<ROL::Vector<double>> design_variables) {
    struct setUpper : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setUpper() : zero_(0) {}
            double apply(const double &/*x*/) const {
                return 0.60;
                return ROL::ROL_INF<double>();
            }
    } setupper;
    struct setLower : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setLower() : zero_(0) {}
            double apply(const double &/*x*/) const {
                return 0.40;
                return -1.0*ROL::ROL_INF<double>();
            }
    } setlower;

    ROL::Ptr<ROL::Vector<double>> l = design_variables->clone();
    ROL::Ptr<ROL::Vector<double>> u = design_variables->clone();

    l->applyUnary(setlower);
    u->applyUnary(setupper);

    return ROL::makePtr<ROL::Bounds<double>>(l,u);
}


template <typename VectorType>
int test(const unsigned int n_des_var)
{
    typedef double RealT;
  
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (mpi_rank == 0) std::cout << std::endl << std::endl;
    if (mpi_rank == 0) std::cout << "Optimization with " << n_des_var << " design variables using VectorType = " << typeid(VectorType).name() << std::endl;
  
    Teuchos::RCP<VectorType>   design_variables_rcp     = Teuchos::rcp(new VectorType);
  
    const dealii::IndexSet locally_owned_dofs = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n_des_var);
    if constexpr (std::is_same_v<VectorType, serial_Vector>) {
        design_variables_rcp->reinit(n_des_var);
    }
    if constexpr (std::is_same_v<VectorType, distributed_Vector>) {

        dealii::IndexSet ghost_dofs(n_des_var);
        for (auto ip = locally_owned_dofs.begin(); ip != locally_owned_dofs.end(); ++ip) {
            const auto index = *ip;

            const auto ghost_low = (index > 0) ? index-1 : 0;
            if (!locally_owned_dofs.is_element(ghost_low)) ghost_dofs.add_index(ghost_low);

            const auto ghost_high = (index < n_des_var-1) ? index+1 : n_des_var-1;
            if (!locally_owned_dofs.is_element(ghost_high)) ghost_dofs.add_index(ghost_high);
            
        }
        design_variables_rcp->reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    }

  
    for (auto xi = design_variables_rcp->begin(); xi != design_variables_rcp->end(); ++xi) {
        *xi = 1.01;
    }
    design_variables_rcp->update_ghost_values();

    dealii::Rol::VectorAdaptor<VectorType> design_variables_rol(design_variables_rcp);
    auto design_variables_ptr = ROL::makePtrFromRef<dealii::Rol::VectorAdaptor<VectorType>>(design_variables_rol);
  
    // Set parameters.
    Teuchos::ParameterList parlist;
    parlist.sublist("Secant").set("Use as Preconditioner", false);
    parlist.sublist("Status Test").set("Gradient Tolerance", 1e-8);
    parlist.sublist("Status Test").set("Iteration Limit", 1000);

    parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
    parlist.sublist("General").sublist("Secant").set("Use as Hessian", USE_BFGS);
    parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-10);
    parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-8);
    parlist.sublist("General").sublist("Krylov").set("Iteration Limit", 1000);
    parlist.sublist("General").sublist("Krylov").set("Use Initial Guess", true);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
    const std::string line_search_curvature = "Null Curvature Condition";
    const std::string line_search_method = "Backtracking";
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


    auto rosenbrock_objective = ROL::makePtr<RosenbrockObjective<VectorType, RealT>>();
    ROL::Ptr<ROL::BoundConstraint<double>>               des_var_bound          = getBoundConstraint(design_variables_ptr);
    std::vector<ROL::Ptr<ROL::Constraint<double>>>       inequality_constraint  = getInequalityConstraint<VectorType,RealT>();
    std::vector<ROL::Ptr<ROL::Vector<double>>>           inequality_dual        = getInequalityMultiplier();
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>>  slack_bound            = getSlackBoundConstraint(design_variables_ptr);
    ROL::OptimizationProblem<RealT> opt_problem = ROL::OptimizationProblem<RealT> ( rosenbrock_objective, design_variables_ptr, des_var_bound,
                                             inequality_constraint, inequality_dual, slack_bound);

  
    // Output stream
    ROL::nullstream bhs; // outputs nothing
    Teuchos::RCP<std::ostream> outStream;
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    ROL::Ptr<const ROL::AlgorithmState<RealT> > algo_state;

    //{
    //    ROL::OptimizationSolver<RealT> opt_solver(opt_problem, parlist);
    //    opt_solver.solve(*outStream);
    //    algo_state = opt_solver.getAlgorithmState()->statusFlag;
    //}

    {
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
    }


  
    Teuchos::RCP<const VectorType> xg = design_variables_rol.getVector();
    *outStream << "The solution to minimization problem is: ";
    std::cout << std::flush;

    //(*xg).print(std::cout);
    for (unsigned int i=0; i<n_des_var; ++i) {
        if (locally_owned_dofs.is_element(i)) {
            std::cout << (*xg)[i] << " ";
        }
        (void) MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << std::flush;
    *outStream << std::endl;

    return algo_state->statusFlag;
}

int main (int argc, char * argv[])
{
    feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = false;
    try {
         test_error += test<distributed_Vector>(2);
         test_error += test<distributed_Vector>(3);
         test_error += test<distributed_Vector>(4);
         test_error += test<distributed_Vector>(5);
         test_error += test<distributed_Vector>(6);
         test_error += test<distributed_Vector>(7);
         // //test_error += test<serial_Vector>(10);
         // test_error += test<distributed_Vector>(10);
         // //test_error += test<serial_Vector>(100);
         // test_error += test<distributed_Vector>(100);
         // //test_error += test<serial_Vector>(101);
         // test_error += test<distributed_Vector>(101);
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

