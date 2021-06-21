// @HEADER
// ***********************************************************************
//
//                           Sacado Package
//                 Copyright (2006) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
// USA
// Questions? Contact David M. Gay (dmgay@sandia.gov) or Eric T. Phipps
// (etphipp@sandia.gov).
//
// ***********************************************************************
// @HEADER

// jacobian_via_dfad_example (extension of dfad_example)
//
//  usage:
//     jacobian_via_dfad_example
//
//  output:
//     prints the results of computing the second derivative a simple function //     with forward nested forward mode AD using the Sacado::Fad::DFad class
//     (uses dynamic memory allocation for number of derivative components).

#include <iostream>
#include <iomanip>

#include "Sacado.hpp"

// The vector function to differentiate
template <typename ScalarT>
std::array<ScalarT,2> func(const std::array<ScalarT,3> & arr, const double &d) {
  std::array<ScalarT,2> val;
  ScalarT a = arr[0]; ScalarT b = arr[1]; ScalarT c = arr[2];
  val[0] = c*std::log(b+1.)/std::sin(a) + d;
  val[1] = c*std::exp((b+3.)*(b+3.))*std::sin(a*a);
  return val;
}

// The analytic derivative of func(a,b,c) with respect to a and b
void func_deriv(double a, double b, double c, double& dfda, double& dfdb, double& dgda, double& dgdb)
{
  dfda = -(c*std::log(b+1.)/std::pow(std::sin(a),2.))*std::cos(a);
  dfdb = c / ((b+1.)*std::sin(a));
  dgda = c*std::exp((b+3.)*(b+3.))*std::cos(a*a)*2.*a;
  dgdb = c*2.*(b+3.)*std::exp((b+3.)*(b+3.))*std::sin(a*a);
}

int main(int /*argc*/, char ** /*argv*/)
{
  int k_last = 2;
  for (int k = 0; k < k_last; k++) {

      double pi = std::atan(1.0)*4.0;

      // Number of independent variables
      int num_deriv = 2;

      // Fad objects
      typedef Sacado::Fad::DFad<double> DFadType;

      // Values of function arguments
      // const DFadType z = 1.0;
      const double a = pi/4;
      const double b = 2.0;
      const double c = 3.0;
      std::array<double,3> varInputs;
      varInputs[0] = a;
      varInputs[1] = b;
      varInputs[2] = c;

      std::array<DFadType,3> args;
      DFadType afad(num_deriv, 0, a);
      DFadType bfad(num_deriv, 1, b);
      DFadType cfad = c;
      args[0] = afad;
      args[1] = bfad;
      args[2] = cfad;
      std::array<DFadType,2> rfad;

      // Compute function
      std::array<double,2> r = func(varInputs,2.0);

      // Compute derivative analytically
      double dfda, dfdb, dgda, dgdb;
      func_deriv(a, b, c, dfda, dfdb, dgda, dgdb);

      // Compute function and derivative with AD
      rfad = func(args,2.0);

      // Extract value and derivatives
      double f_ad    = rfad[0].val(); // f
      double g_ad    = rfad[1].val(); // g
      double dfda_ad = rfad[0].dx(0); // df/da
      double dfdb_ad = rfad[0].dx(1); // df/db
      double dgda_ad = rfad[1].dx(0); // dg/da
      double dgdb_ad = rfad[1].dx(1); // dg/db

      if (k == k_last-1) {
      // Print the results
      int p = 4;
      int w = p+7;
      std::cout.setf(std::ios::scientific);
      std::cout.precision(p);
      std::cout << "        f = " << std::setw(w) << r[0] << " (original) == "
                << std::setw(w) << f_ad << " (AD) Error = " << std::setw(w)
                << r[0] - f_ad << std::endl

                << "        g = " << std::setw(w) << r[1] << " (original) == "
                << std::setw(w) << g_ad << " (AD) Error = " << std::setw(w)
                << r[1] - g_ad << std::endl

                << "    df/da = " << std::setw(w) << dfda << " (analytic) == "
                << std::setw(w) << dfda_ad << " (AD) Error = " << std::setw(w)
                << dfda - dfda_ad << std::endl

                << "    df/db = " << std::setw(w) << dfdb << " (analytic) == "
                << std::setw(w) << dfdb_ad << " (AD) Error = " << std::setw(w)
                << dfdb - dfdb_ad << std::endl

                << "    dg/da = " << std::setw(w) << dgda << " (analytic) == "
                << std::setw(w) << dgda_ad << " (AD) Error = " << std::setw(w)
                << dgda - dgda_ad << std::endl

                << "    dg/db = " << std::setw(w) << dgdb << " (analytic) == "
                << std::setw(w) << dgdb_ad << " (AD) Error = " << std::setw(w)
                << dgdb - dgdb_ad << std::endl;
       }

       double tol = 1.0e-14;
       if (std::fabs(r[0] - f_ad)          < tol &&
           std::fabs(r[1] - g_ad)          < tol &&
           std::fabs(dfda - dfda_ad)       < tol &&
           std::fabs(dfdb - dfdb_ad)       < tol &&
           std::fabs(dgda - dgda_ad)       < tol &&
           std::fabs(dgdb - dgdb_ad)       < tol) {
         std::cout << "\nExample passed!" << std::endl;
      //   return 0;
       }
      else {
        std::cout <<"\nSomething is wrong, example failed!" << std::endl;
      //   return 1;
      }
    }
    return 0;
}
