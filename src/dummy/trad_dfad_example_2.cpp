// $Id$ 
// $Source$ 
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

// trad_dfad_example
//
//  usage: 
//     trad_dfad_example
//
//  output:  
//     prints the results of computing the second derivative a simple function //     with forward nested forward and reverse mode AD using the 
//     Sacado::Fad::DFad and Sacado::Rad::ADvar classes.

#include <iostream>
#include <iomanip>

#include "Sacado.hpp"

// The function to differentiate
template <typename ScalarT>
ScalarT func(const ScalarT& a) {
  ScalarT r = a*a*a;
  return r;
}

// The analytic derivative of func(a,b,c) with respect to a and b
void func_deriv(double a, double &drda)
{
  drda = 3*a*a;
}

// The analytic second derivative of func(a,b,c) with respect to a and b
void func_deriv2(double a, double &d2rda2)
{
  d2rda2 = 6*a;
}

int main(int /*argc*/, char ** /*argv*/)
{
  double r_ad, drda_ad, d2rda2_ad;
  double r, drda, d2rda2;
  int k_last = 2;
  for (int k = 0; k < k_last; k++) {

      // Values of function arguments
      double a = 1./4;

      // Number of independent variables
      int num_deriv = 1;

      Sacado::Fad::DFad<double> a_temp(num_deriv, 0, a);
      // Fad objects
      Sacado::Rad::ADvar< Sacado::Fad::DFad<double> > arad;
      arad = a_temp;
      // arad = a;
      // arad.adj().val() = 1.0;
      // arad.val().diff(0,1);
      // arad.adj().diff(0,1);
      Sacado::Rad::ADvar< Sacado::Fad::DFad<double> > rrad;

      // Compute function
      r = func(a);

      // Compute derivative analytically
      func_deriv(a, drda);
      // Compute second derivative analytically
      func_deriv2(a, d2rda2);

      // Compute function and derivative with AD
      rrad = func(arad);

      Sacado::Rad::ADvar< Sacado::Fad::DFad<double> >::Gradcomp();

      // Extract value and derivatives
      r_ad = rrad.val().val();       // r
      drda_ad = arad.adj().val();    // dr/da
      d2rda2_ad = arad.adj().dx(0);  // d^2r/da^2
      //Sacado::Rad::ADvar< Sacado::Fad::DFad<double> >:: aval_reset ();
      //Sacado::Rad::ADcontext< Sacado::Fad::DFad<double> >:: re_init();

      //if (k == k_last - 1) {
          // Print the results
          int p = 4;
          int w = p+7;
          std::cout.setf(std::ios::scientific);
          std::cout.precision(p);
          std::cout << "        r = " << std::setw(w) << r << " (original) == " 
                << std::setw(w) << r_ad << " (AD) Error = " << std::setw(w) 
                << r - r_ad << std::endl
                << "    dr/da = " << std::setw(w) << drda << " (analytic) == " 
                << std::setw(w) << drda_ad << " (AD) Error = " << std::setw(w) 
                << drda - drda_ad << std::endl
                << "d^2r/da^2 = " << std::setw(w) << d2rda2 << " (analytic) == " 
                << std::setw(w) << d2rda2_ad << " (AD) Error = " << std::setw(w) 
                << d2rda2 - d2rda2_ad << std::endl
                << std::endl;
      //}
  }

  double tol = 1.0e-14;
  if (std::fabs(r - r_ad)             < tol &&
      std::fabs(drda - drda_ad)       < tol &&
      std::fabs(d2rda2 - d2rda2_ad)   < tol) {
    std::cout << "\nExample passed!" << std::endl;
    return 0;
  }
  else {
    std::cout <<"\nSomething is wrong, example failed!" << std::endl;
    return 1;
  }
}

