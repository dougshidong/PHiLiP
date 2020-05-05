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

#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Objective.hpp"
#include "ROL_StatusTest.hpp"
#include "Teuchos_GlobalMPISession.hpp"

// This test is used to check that the dealii::LinearAlgebra::distributed::Vector<double>
// is working properly with ROL. This is done by performing an unconstrained optimization
// of the Rosenbrock function.
using serial_Vector = typename dealii::Vector<double>;
using distributed_Vector = typename dealii::LinearAlgebra::distributed::Vector<double>;
// Use ROL to minimize the objective function, f(x,y) = x^2 + y^2.

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
      // const double tpi = 2* std::atan(1.0)*4;
      // return std::sin((*xp)[0]*tpi) + std::sin((*xp)[1]*tpi);
      const Real rosenbrock = dealii::Utilities::MPI::sum(local_rosenbrock, MPI_COMM_WORLD);
      return rosenbrock;
    }

    /// Return the Rosenbrock objective gradient.
    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
      using ADtype = Sacado::Fad::DFad<double>;


      Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
      Teuchos::RCP<VectorType>       gp = this->get_rcp_to_VectorType(g);

      // gp->reinit(xp->get_partitioner());
      // gp->set_ghost_state(true);

      // const unsigned int nx = (*xp).size();
      // (*gp) *= 0.0;
      // for (unsigned int i = 0; i < nx-1; ++i) {
      //     const Real &x1 = (*xp)[i];
      //     const Real &x2 = (*xp)[i+1];
      //     // https://www.wolframalpha.com/input/?i=f%28a%2Cb%29+%3D+100*%28b-a*a%29%5E2+%2B+%281-a%29%5E2%2C+df%2Fda
      //      const Real drosenbrock_dx1 = 2.0*(200*x1*x1*x1 - 200*x1*x2 + x1 - 1.0);
      //     (*gp)[i]   += drosenbrock_dx1;

      //     // https://www.wolframalpha.com/input/?i=f%28a%2Cb%29+%3D+100*%28b-a*a%29%5E2+%2B+%281-a%29%5E2%2C+df%2Fdb
      //     const Real drosenbrock_dx2 = 200.0*(x2-x1*x1);
      //     (*gp)[i+1] += drosenbrock_dx2;
      // }

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
      // std::cout << (*gp).l2_norm() << std::endl;
      // (*gp).print(std::cout);
    }
};

template <typename VectorType>
int test(const unsigned int n_des_var)
{
    typedef double RealT;
  
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    if (mpi_rank == 0) std::cout << std::endl << "**************************************************" << std::endl;
    if (mpi_rank == 0) std::cout << "Optimization with " << n_des_var << " using VectorType = " << typeid(VectorType).name() << std::endl;
  
    Teuchos::RCP<VectorType>   x_rcp     = Teuchos::rcp(new VectorType);
  
    dealii::IndexSet locally_owned_dofs(n_des_var);
    if constexpr (std::is_same_v<VectorType, serial_Vector>) {
        x_rcp->reinit(n_des_var);
        locally_owned_dofs.add_range(0, n_des_var);
    }
    if constexpr (std::is_same_v<VectorType, distributed_Vector>) {
        const unsigned int n_dofs_per_proc = n_des_var / n_mpi;
        const unsigned int low_i = n_dofs_per_proc * mpi_rank;
        const unsigned int high_i = (n_mpi-1) == mpi_rank ? n_des_var : n_dofs_per_proc * (mpi_rank+1);
        locally_owned_dofs.add_range(low_i, high_i);

        dealii::IndexSet ghost_dofs(n_des_var);
        for (auto ip = locally_owned_dofs.begin(); ip != locally_owned_dofs.end(); ++ip) {
            const auto index = *ip;

            const auto ghost_low = (index > 0) ? index-1 : 0;
            if (!locally_owned_dofs.is_element(ghost_low)) ghost_dofs.add_index(ghost_low);

            const auto ghost_high = (index < n_des_var-1) ? index+1 : n_des_var-1;
            if (!locally_owned_dofs.is_element(ghost_high)) ghost_dofs.add_index(ghost_high);
            
        }
        x_rcp->reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    }

  
    for (auto xi = x_rcp->begin(); xi != x_rcp->end(); ++xi) {
        *xi = 0.0;
    }
    x_rcp->update_ghost_values();

    dealii::Rol::VectorAdaptor<VectorType> x_rol(x_rcp);
  
    RosenbrockObjective<VectorType, RealT> rosenbrock_objective;
  
    // Set parameters.
    Teuchos::ParameterList parlist;
    parlist.sublist("Secant").set("Use as Preconditioner", false);
    parlist.sublist("Status Test").set("Gradient Tolerance", 1e-10);
    parlist.sublist("Status Test").set("Iteration Limit", 1000);

    // Define algorithm.
    ROL::Algorithm<RealT> algo("Line Search", parlist);
  
    // Output stream
    ROL::nullstream bhs; // outputs nothing
    Teuchos::RCP<std::ostream> outStream;
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    // Run Algorithm
    algo.run(x_rol, rosenbrock_objective, true, *outStream);
  
    Teuchos::RCP<const VectorType> xg = x_rol.getVector();
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

    return algo.getState()->statusFlag;
}

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = false;
    try {
         test_error += test<serial_Vector>(10);
         test_error += test<distributed_Vector>(10);
         test_error += test<serial_Vector>(100);
         test_error += test<distributed_Vector>(100);
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
