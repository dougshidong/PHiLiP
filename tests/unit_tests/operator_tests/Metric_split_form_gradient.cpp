#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>

#include <deal.II/base/convergence_table.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include "dg/dg_factory.hpp"
#include "operators/operators.h"
//#include <GCL_test.h>

const double TOLERANCE = 1E-6;
using namespace std;
//namespace PHiLiP {


template <int dim>
class CurvManifold: public dealii::ChartManifold<dim,dim,dim> {
    virtual dealii::Point<dim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<dim> push_forward(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,dim,dim> push_forward_gradient(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.
};
template<int dim>
dealii::Point<dim> CurvManifold<dim>::pull_back(const dealii::Point<dim> &space_point) const 
{
    using namespace PHiLiP;
    const double pi = atan(1)*4.0;
    dealii::Point<dim> x_ref;
    dealii::Point<dim> x_phys;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = space_point[idim];
        x_phys[idim] = space_point[idim];
    }
    dealii::Vector<double> function(dim);
    dealii::FullMatrix<double> derivative(dim);
    double beta =1.0/10.0;
    double alpha =1.0/10.0;
    int flag =0;
    while(flag != dim){
#if 0
        for(int idim=0;idim<dim;idim++){
            function[idim] = 1.0/20.0; 
            for(int idim2=0;idim2<dim;idim2++){
                function[idim] *= std::cos(2.0 * pi* x_ref[idim2]);
            }
            function[idim] += x_ref[idim] - x_phys[idim];
        }
#endif

//#if 0
    if(dim==2){
        function[0] = x_ref[0] - x_phys[0] +beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function[1] = x_ref[1] - x_phys[1] +beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else{
    #if 0
        function[0] = x_ref[0] - x_phys[0] +alpha*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        function[1] = x_ref[1] - x_phys[1] +alpha*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        function[2] = x_ref[2] - x_phys[2] +alpha*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
    #endif
    //heavily warped
   // #if 0
      // function[0] = x_ref[0] - x_phys[0] +alpha*std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]);
      // function[1] = x_ref[1] - x_phys[1] +alpha*exp(1.0-x_ref[1])*std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]);
      // double x_temp = x_ref[0] +alpha*std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]);
      // double y_temp = x_ref[1] +alpha*exp(1.0-x_ref[1])*std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]);
       // function[2] = x_ref[2] - x_phys[2] +1.0/20.0*( std::sin(2.0 * pi * x_temp) + std::sin(2.0 * pi * y_temp));
   // #endif
 //  #if 0
        function[0] = x_ref[0] - x_phys[0] +alpha*(std::cos(pi * x_ref[2]) + std::cos(pi * x_ref[1]));
        function[1] = x_ref[1] - x_phys[1] +alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) + std::sin(pi* x_ref[2]));
        function[2] = x_ref[2] - x_phys[2] +1.0/20.0*( std::sin(2.0 * pi * x_ref[0]) + std::sin(2.0 * pi * x_ref[1]));
  //  #endif
    }
//#endif


    #if 0
        for(int idim=0; idim<dim; idim++){
            for(int idim2=0; idim2<dim;idim2++){
                derivative[idim][idim2] = - 1.0/20.0*2.0 * pi;
                for(int idim3 =0;idim3<dim; idim3++){
                    if(idim2 == idim3)
                        derivative[idim][idim2] *=std::sin(2.0 * pi * x_ref[idim3]);
                    else
                        derivative[idim][idim2] *=std::cos(2.0 * pi* x_ref[idim3]);
                }
                if(idim == idim2)
                    derivative[idim][idim2] += 1.0;
            }
        }
#endif
//#if 0
    if(dim==2){
        derivative[0][0] = 1.0 - beta* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        derivative[0][1] =  - beta*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        derivative[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        derivative[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
    }
    else{
    #if 0
        derivative[0][0] = 1.0  - alpha* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        derivative[0][1] =      - alpha*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        derivative[0][2] =        alpha*2.0*pi * std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(2.0*pi*(x_ref[2]));

        derivative[1][0] =       alpha*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        derivative[1][1] =  1.0 -alpha*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));  
        derivative[1][2] =       alpha*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::cos(3.0*pi/2.0*(x_ref[2]));

        derivative[2][0] =       alpha*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        derivative[2][1] =     - alpha*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        derivative[2][2] = 1.0 + alpha*5.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(5.0*pi/2.0*(x_ref[2]));
    #endif
    //heavily warped
   // #if 0
     //  derivative[0][0] = 1.0 + alpha*pi*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
     //  derivative[0][1] =       alpha*pi*std::cos(pi*x_ref[1])*std::sin(pi*x_ref[0]);
     //  derivative[0][2] =  0.0;

     //  derivative[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
     //  derivative[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1])
     //                          +alpha*pi*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::cos(pi*x_ref[1]);  
     //  derivative[1][2] =  0.0;

      //  double x_temp = x_ref[0] + alpha*std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]);
     //  double y_temp = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]);
     //  derivative[2][0] = 1.0/10.0*pi*std::cos(2.0*pi*x_temp)*derivative[0][0] 
     //                    +1.0/10.0*pi*std::cos(2.0*pi*y_temp)*derivative[1][0];
     //  derivative[2][1] = 1.0/10.0*pi*std::cos(2.0*pi*x_temp)*derivative[0][1]
     //                    +1.0/10.0*pi*std::cos(2.0*pi*y_temp)*derivative[1][1];
     //
     //
   //  #if 0
        derivative[0][0] = 1.0;
        derivative[0][1] =      - alpha*pi*std::sin(pi*x_ref[1]);
        derivative[0][2] =   - alpha*pi*std::sin(pi*x_ref[2]);

        derivative[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0]);
        derivative[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[2]));  
        derivative[1][2] =  alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[2]);
        derivative[2][0] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[0]);
        derivative[2][1] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[1]);
        derivative[2][2] = 1.0;
   // #endif
   // #endif
    }
//#endif

        dealii::FullMatrix<double> Jacobian_inv(dim);
        Jacobian_inv.invert(derivative);
        dealii::Vector<double> Newton_Step(dim);
        Jacobian_inv.vmult(Newton_Step, function);
        for(int idim=0; idim<dim; idim++){
            x_ref[idim] -= Newton_Step[idim];
        }
        flag=0;
        for(int idim=0; idim<dim; idim++){
            if(std::abs(function[idim]) < 1e-15)
                flag++;
        }
        if(flag == dim)
            break;
    }
    std::vector<double> function_check(dim);
#if 0
    for(int idim=0;idim<dim; idim++){
        function_check[idim] = 1.0/20.0;
        for(int idim2=0; idim2<dim; idim2++){
            function_check[idim] *= std::cos(2.0 * pi * x_ref[idim2]);
        }
        function_check[idim] += x_ref[idim];
    }
#endif
//#if 0
    if(dim==2){
        function_check[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function_check[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else{
    #if 0
        function_check[0] = x_ref[0] + alpha*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        function_check[1] = x_ref[1] + alpha*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        function_check[2] = x_ref[2] + alpha*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
    #endif
    //heavily warped
    //#if 0
      // function_check[0] = x_ref[0] +alpha*std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]);
      // function_check[1] = x_ref[1] +alpha*exp(1.0-x_ref[1])*std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]);
      //  double x_temp = x_ref[0] + alpha*std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]);
      // double y_temp = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]);
      // function_check[2] = x_ref[2] +1.0/20.0*( std::sin(2.0 * pi * x_temp) + std::sin(2.0 * pi * y_temp));
   // #endif
 //  #if 0
        function_check[0] = x_ref[0] +alpha*(std::cos(pi * x_ref[2]) + std::cos(pi * x_ref[1]));
        function_check[1] = x_ref[1] +alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) + std::sin(pi* x_ref[2]));
        function_check[2] = x_ref[2] +1.0/20.0*( std::sin(2.0 * pi * x_ref[0]) + std::sin(2.0 * pi * x_ref[1]));
 //   #endif
    }
//#endif
    std::vector<double> error(dim);
    for(int idim=0; idim<dim; idim++) 
        error[idim] = std::abs(function_check[idim] - x_phys[idim]);
    if (error[0] > 1e-13) {
        std::cout << "Large error " << error[0] << std::endl;
        for(int idim=0;idim<dim; idim++)
        std::cout << "dim " << idim << " xref " << x_ref[idim] <<  " x_phys " << x_phys[idim] << " function Check  " << function_check[idim] << " Error " << error[idim] << " Flag " << flag << std::endl;
    }

    return x_ref;

}

template<int dim>
dealii::Point<dim> CurvManifold<dim>::push_forward(const dealii::Point<dim> &chart_point) const 
{
    const double pi = atan(1)*4.0;

    dealii::Point<dim> x_ref;
    dealii::Point<dim> x_phys;
    for(int idim=0; idim<dim; idim++)
        x_ref[idim] = chart_point[idim];
#if 0
    for(int idim=0; idim<dim; idim++){
        x_phys[idim] = 1.0/20.0;
        for(int idim2=0;idim2<dim; idim2++){
           x_phys[idim] *= std::cos( 2.0 * pi * x_ref[idim2]);
        }
        x_phys[idim] += x_ref[idim];
    }
#endif
    double beta = 1.0/10.0;
    double alpha = 1.0/10.0;
//#if 0
    if(dim==2){
        x_phys[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        x_phys[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else{
    #if 0
        x_phys[0] = x_ref[0] + alpha*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        x_phys[1] = x_ref[1] + alpha*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        x_phys[2] = x_ref[2] + alpha*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
    #endif
    //heavily warped
    //#if 0
      // x_phys[0] =x_ref[0] +  alpha*std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]);
      // x_phys[1] =x_ref[1] +  alpha*exp(1.0-x_ref[1])*std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]);
       // x_phys[2] =x_ref[2] +  1.0/20.0*( std::sin(2.0 * pi * x_phys[0]) + std::sin(2.0 * pi * x_phys[1]));
    //#endif
   // #if 0
        x_phys[0] =x_ref[0] +  alpha*(std::cos(pi * x_ref[2]) + std::cos(pi * x_ref[1]));
        x_phys[1] =x_ref[1] +  alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) + std::sin(pi* x_ref[2]));
        x_phys[2] =x_ref[2] +  1.0/20.0*( std::sin(2.0 * pi * x_ref[0]) + std::sin(2.0 * pi * x_ref[1]));
  //  #endif
    }
//#endif
    return dealii::Point<dim> (x_phys); // Trigonometric
}

template<int dim>
dealii::DerivativeForm<1,dim,dim> CurvManifold<dim>::push_forward_gradient(const dealii::Point<dim> &chart_point) const 
{
    const double pi = atan(1)*4.0;
    dealii::DerivativeForm<1, dim, dim> dphys_dref;
#if 0
    dealii::Point<dim> x;
    for(int idim=0; idim<dim; idim++)
        x[idim] = chart_point[idim];
    for(int idim=0; idim<dim; idim++){
        for(int idim2=0; idim2<dim;idim2++){
            dphys_dref[idim][idim2] = - 1.0/20.0*2.0 * pi;
            for(int idim3 =0;idim3<dim; idim3++){
                if(idim2 == idim3)
                    dphys_dref[idim][idim2] *=std::sin(2.0 * pi * x[idim3]);
                else
                     dphys_dref[idim][idim2] *=std::cos(2.0 * pi* x[idim3]);
            }     
            if(idim == idim2)
                dphys_dref[idim][idim2] += 1.0;
        }
    }
#endif
//#if 0
    double beta = 1.0/10.0;
    double alpha = 1.0/10.0;
    dealii::Point<dim> x_ref;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = chart_point[idim];
    }

    if(dim==2){
        dphys_dref[0][0] = 1.0 - beta*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        dphys_dref[0][1] =  - beta*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        dphys_dref[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        dphys_dref[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
    }
    else{
    #if 0
        dphys_dref[0][0] = 1.0 - alpha*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        dphys_dref[0][1] =     - alpha*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        dphys_dref[0][2] =       alpha*2.0*pi * std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(2.0*pi*(x_ref[2]));

        dphys_dref[1][0] =       alpha*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        dphys_dref[1][1] =  1.0 -alpha*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));  
        dphys_dref[1][2] =       alpha*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::cos(3.0*pi/2.0*(x_ref[2]));

        dphys_dref[2][0] =       alpha*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        dphys_dref[2][1] =     - alpha*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        dphys_dref[2][2] = 1.0 + alpha*5.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(5.0*pi/2.0*(x_ref[2]));
    #endif
    //heavily warped
   // #if 0
     //  dphys_dref[0][0] = 1.0 + alpha*pi*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
     //  dphys_dref[0][1] =       alpha*pi*std::cos(pi*x_ref[1])*std::sin(pi*x_ref[0]);
     //  dphys_dref[0][2] =  0.0;

     //  dphys_dref[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
     //  dphys_dref[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1])
     //                          +alpha*pi*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::cos(pi*x_ref[1]);  
     //  dphys_dref[1][2] =  0.0;

      //  double x_phys = x_ref[0] + alpha*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]);
     //  double y_phys = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]);
     //  dphys_dref[2][0] = 1.0/10.0*pi*std::cos(2.0*pi*x_phys)*dphys_dref[0][0] 
     //                    +1.0/10.0*pi*std::cos(2.0*pi*y_phys)*dphys_dref[1][0];
     //  dphys_dref[2][1] = 1.0/10.0*pi*std::cos(2.0*pi*x_phys)*dphys_dref[0][1]
     //                    +1.0/10.0*pi*std::cos(2.0*pi*y_phys)*dphys_dref[1][1];
    //#endif
    //#if 0
        dphys_dref[0][0] = 1.0;
        dphys_dref[0][1] =      - alpha*pi*std::sin(pi*x_ref[1]);
        dphys_dref[0][2] =   - alpha*pi*std::sin(pi*x_ref[2]);

        dphys_dref[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0]);
        dphys_dref[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[2]));  
        dphys_dref[1][2] =  alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[2]);
        dphys_dref[2][0] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[0]);
        dphys_dref[2][1] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[1]);
        dphys_dref[2][2] = 1.0;
   // #endif
    }
//#endif

    return dphys_dref;
}

template<int dim>
std::unique_ptr<dealii::Manifold<dim,dim> > CurvManifold<dim>::clone() const 
{
    return std::make_unique<CurvManifold<dim>>();
}

template <int dim>
static dealii::Point<dim> warp (const dealii::Point<dim> &p)
{
    const double pi = atan(1)*4.0;
    dealii::Point<dim> q = p;

    double beta =1.0/10.0;
    double alpha =1.0/10.0;
    if (dim == 2){
        q[dim-2] =p[dim-2] +  beta*std::cos(pi/2.0 * p[dim-2]) * std::cos(3.0 * pi/2.0 * p[dim-1]);
        q[dim-1] =p[dim-1] +  beta*std::sin(2.0 * pi * (p[dim-2])) * std::cos(pi /2.0 * p[dim-1]);
    }
    if(dim==3){
    #if 0
        q[0] =p[0] +  alpha*std::cos(pi/2.0 * p[0]) * std::cos(3.0 * pi/2.0 * p[1]) * std::sin(2.0 * pi * (p[2]));
        q[1] =p[1] +  alpha*std::sin(2.0 * pi * (p[0])) * std::cos(pi /2.0 * p[1]) * std::sin(3.0 * pi /2.0 * p[2]);
        q[2] =p[2] +  alpha*std::sin(2.0 * pi * (p[0])) * std::cos(3.0 * pi/2.0 * p[1]) * std::cos(5.0 * pi/2.0 * p[2]);
    #endif
    //heavily warped
   // #if 0
       // q[0] =p[0] +  alpha*std::sin(pi * p[0]) * std::sin(pi * p[1]);
       // q[1] =p[1] +  alpha*exp(1.0-p[1])*std::sin(pi * p[0]) * std::sin(pi* p[1]);
       // q[2] =p[2] +  1.0/20.0*( std::sin(2.0 * pi * q[0]) + std::sin(2.0 * pi * q[1]));
   // #endif
  // #if 0
        q[0] =p[0] +  alpha*(std::cos(pi * p[2]) + std::cos(pi * p[1]));
        q[1] =p[1] +  alpha*exp(1.0-p[1])*(std::sin(pi * p[0]) + std::sin(pi* p[2]));
        q[2] =p[2] +  1.0/20.0*( std::sin(2.0 * pi * p[0]) + std::sin(2.0 * pi * p[1]));
  //  #endif
    }

    return q;
}

/****************************
 * End of Curvilinear Grid
 * ***************************/

int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);

   // all_parameters_new.use_collocated_nodes=true;
    all_parameters_new.use_curvilinear_split_form=true;
    const int dim_check = 1;

    //unsigned int poly_degree = 3;
    double left = 0.0;
    double right = 1.0;
    const bool colorize = true;
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = 2;
    const unsigned int n_grids = 5;
    std::array<double,n_grids> grid_size;
    std::array<double,n_grids> soln_error;
    std::array<double,n_grids> soln_error_inf;
    for(unsigned int poly_degree = 2; poly_degree<5; poly_degree++){
        unsigned int grid_degree = poly_degree;
    for(unsigned int igrid=igrid_start; igrid<n_grids; ++igrid){
pcout<<" Grid Index"<<igrid<<std::endl;
    //Generate a standard grid

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
        dealii::GridGenerator::hyper_cube (*grid, left, right, colorize);
        grid->refine_global(igrid);
pcout<<" made grid for Index"<<igrid<<std::endl;

//Warp the grid
//IF WANT NON-WARPED GRID COMMENT UNTIL SAYS "NOT COMMENT"
//#if 0
    dealii::GridTools::transform (&warp<dim>, *grid);

// Assign a manifold to have curved geometry
    const CurvManifold<dim> curv_manifold;
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid->reset_all_manifolds();
    grid->set_all_manifold_ids(manifold_id);
    grid->set_manifold ( manifold_id, curv_manifold );
//#endif
//"END COMMENT" TO NOT WARP GRID

    //setup operator
    OPERATOR::OperatorBase<dim,real> operators(&all_parameters_new, nstate, poly_degree, poly_degree, grid_degree); 
//setup DG
   // std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, grid);
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();
    

            dealii::IndexSet locally_owned_dofs;
            dealii::IndexSet ghost_dofs;
            dealii::IndexSet locally_relevant_dofs;
            locally_owned_dofs = dg->dof_handler.locally_owned_dofs();
            dealii::DoFTools::extract_locally_relevant_dofs(dg->dof_handler, ghost_dofs);
            locally_relevant_dofs = ghost_dofs;
            ghost_dofs.subtract_set(locally_owned_dofs);
          //  dealii::LinearAlgebra::distributed::Vector<double> solution;
          //  solution.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
            dealii::LinearAlgebra::distributed::Vector<double> solution_deriv;
            solution_deriv.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
//Interpolate IC
#if 0
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
            const dealii::MappingQGeneric<dim, dim> mapping_collection3 (poly_degree+1);
            dealii::FEValues<dim,dim> fe_values_vol(mapping_collection3, dg->fe_collection[0], dg->volume_quadrature_collection[0], 
                                dealii::update_values | dealii::update_JxW_values | 
                                dealii::update_jacobians |  
                                dealii::update_quadrature_points | dealii::update_inverse_jacobians);
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
pcout<<"issue here"<<std::endl;
                fe_values_vol.reinit (current_cell);
pcout<<"knew it"<<std::endl;
                current_dofs_indices.resize(n_dofs_cell);
                current_cell->get_dof_indices (current_dofs_indices);
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    solution[current_dofs_indices[idof]]=0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        const dealii::Point<dim> qpoint = (fe_values_vol.quadrature_point(iquad));
                        double exact = 1.0;
                       for (int idim=0; idim<dim; idim++){
                                exact *= exp(-(qpoint[idim])*(qpoint[idim]));
                        }
                        solution[current_dofs_indices[idof]] +=operators.vol_projection_operator[poly_degree][idof][iquad] *exact; 
                    }   
                }
        }

#endif
//setup metric and solve

    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts      = dg->volume_quadrature_collection[poly_degree].size();

            const dealii::FESystem<dim> &fe_metric = (dg->high_order_grid->fe_system);
            const unsigned int n_metric_dofs = fe_metric.dofs_per_cell; 
            auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
                if (!current_cell->is_locally_owned()) continue;
	
//pcout<<"grid degree "<<grid_degree<<" metric dofs "<<n_metric_dofs<<std::endl;
                std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs);
                metric_cell->get_dof_indices (current_metric_dofs_indices);
                std::vector<std::vector<real>> mapping_support_points(dim);
                std::vector<std::vector<real>> phys_quad_pts(dim);
                for(int idim=0; idim<dim; idim++){
                    mapping_support_points[idim].resize(n_metric_dofs/dim);
                    phys_quad_pts[idim].resize(n_quad_pts);
                }
#if 0
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                    const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                    const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
                    mapping_support_points[istate][ishape] = val; 
                }
#endif
                dealii::QGaussLobatto<dim> vol_GLL(grid_degree +1);
                for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
                    for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                        const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                        const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                        mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
                    }
                }
                std::vector<dealii::FullMatrix<real>> metric_cofactor(n_quad_pts);
                std::vector<real> determinant_Jacobian(n_quad_pts);
                for(unsigned int iquad=0;iquad<n_quad_pts; iquad++){
                    metric_cofactor[iquad].reinit(dim, dim);
                }
                operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts, n_metric_dofs/dim, mapping_support_points, determinant_Jacobian, metric_cofactor);

                //get physical split grdient in covariant basis
                std::vector<std::vector<dealii::FullMatrix<real>>> physical_gradient(nstate);
                for(unsigned int istate=0; istate<nstate; istate++){
                    physical_gradient[istate].resize(dim);
                    for(int idim=0; idim<dim; idim++){
                        physical_gradient[istate][idim].reinit(n_quad_pts, n_quad_pts);    
                    }
                }
                operators.get_Jacobian_scaled_physical_gradient(true, operators.gradient_flux_basis[poly_degree], metric_cofactor, n_quad_pts, nstate, physical_gradient); 

            //interpolate solution
                current_dofs_indices.resize(n_dofs_cell);
                current_cell->get_dof_indices (current_dofs_indices);
                for(int idim=0; idim<dim; idim++){
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        phys_quad_pts[idim][iquad] = 0.0;
                        for(unsigned int jquad=0; jquad<n_metric_dofs/dim; jquad++){
                            phys_quad_pts[idim][iquad] += operators.mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][iquad][jquad]
                                                        * mapping_support_points[idim][jquad];
                        }
                    }
                 //   operators.mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree].vmult(phys_quad_pts[idim], mapping_support_points[idim]);
                }
                std::vector<real> soln(n_quad_pts);
               // for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                   // solution[current_dofs_indices[idof]]=0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    //    soln[iquad] = 0.0;
                      //  const dealii::Point<dim> qpoint = (fe_values_vol.quadrature_point(iquad));
                        double exact = 1.0;
                       for (int idim=0; idim<dim; idim++){
                               // exact *= exp(-(qpoint[idim])*(qpoint[idim]));
                                exact *= exp(-(phys_quad_pts[idim][iquad])*(phys_quad_pts[idim][iquad]));
                        }
                        soln[iquad] = exact;
                    //    solution[current_dofs_indices[idof]] +=operators.vol_projection_operator[poly_degree][idof][iquad] *exact; 
                    }   
               // }
                //end interpolated solution


                dealii::Vector<real> soln_derivative_x(n_quad_pts);
                for(int istate=0; istate<nstate; istate++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                   // solution_deriv[current_dofs_indices[iquad]] = 0.0;
                    soln_derivative_x[iquad]=0.0;
                    for(unsigned int idof=0; idof<n_quad_pts; idof++){
                        soln_derivative_x[iquad] += physical_gradient[istate][dim_check][iquad][idof] * soln[idof];
                       // soln_derivative_x[iquad] += physical_gradient[istate][0][iquad][idof] * solution[current_dofs_indices[idof]] / determinant_Jacobian[iquad];
                       // solution_deriv[current_dofs_indices[iquad]] += physical_gradient[istate][0][iquad][idof] * solution[current_dofs_indices[idof]] / determinant_Jacobian[iquad];
                    }
                    soln_derivative_x[iquad] /= determinant_Jacobian[iquad];
                }
                 
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    solution_deriv[current_dofs_indices[idof]] = 0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        solution_deriv[current_dofs_indices[idof]] += operators.vol_projection_operator[poly_degree][idof][iquad]
                                                                    * soln_derivative_x[iquad];
                    }
//pcout<<" proj "<<solution_deriv[current_dofs_indices[idof]]<<" other "<<soln_derivative_x[idof]<<std::endl;
                }
                }




            }

    //TEST ERROR OOA

pcout<<"OOA here"<<std::endl;
            double l2error = 0.0;
            double linf_error = 0.0;
            int overintegrate = 4;
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
       //     const dealii::MappingQGeneric<dim, dim> mapping_collection2 (poly_degree+1);
           // dealii::FEValues<dim,dim> fe_values_extra(mapping_collection2, dg->fe_collection[0], quad_extra, 
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                                dealii::update_values | dealii::update_JxW_values | 
                                dealii::update_jacobians |  
                                dealii::update_quadrature_points | dealii::update_inverse_jacobians);
            const unsigned int n_quad_pts_extra = fe_values_extra.n_quadrature_points;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            dealii::Vector<real> soln_at_q(n_quad_pts_extra);
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
                fe_values_extra.reinit(current_cell);
                dofs_indices.resize(fe_values_extra.dofs_per_cell);
                current_cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts_extra; ++iquad) {
                    soln_at_q[iquad] = 0.0;
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        soln_at_q[iquad] += solution_deriv[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, 0);
                    }
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact_x=1.0;
                    for(int idim=0; idim<dim; idim++){
                        uexact_x *= exp(-((qpoint[idim]) * (qpoint[idim])));
                    }
                    uexact_x *= - 2.0 * qpoint[dim_check];
//pcout<<" soln "<<soln_at_q[iquad]<<" exact "<<uexact_x<<std::endl;
                    l2error += pow(soln_at_q[iquad] - uexact_x, 2) * fe_values_extra.JxW(iquad);
                    double inf_temp = std::abs(soln_at_q[iquad]-uexact_x);
                    if(inf_temp > linf_error){
                        linf_error = inf_temp;
                    }
                }

            }
pcout<<"got OOA here"<<std::endl;


    const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, MPI_COMM_WORLD));
            const double linferror_mpi= (dealii::Utilities::MPI::max(linf_error, MPI_COMM_WORLD));
            // Convergence table
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
            soln_error_inf[igrid] = linferror_mpi;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("soln_Linf_error", linferror_mpi);


            pcout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                 << " Linf-soln_error: " << linferror_mpi
                 << std::endl;


            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                const double slope_soln_err_inf = log(soln_error_inf[igrid]/soln_error_inf[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << soln_error[igrid-1]
                     << "  solution_error2 " << soln_error[igrid]
                     << "  slope " << slope_soln_err
                     << "  solution_error1_inf " << soln_error_inf[igrid-1]
                     << "  solution_error2_inf " << soln_error_inf[igrid]
                     << "  slope " << slope_soln_err_inf
                     << std::endl;
            }


    //end test error OOA



    }//end grid refinement loop

    const int igrid = n_grids-1;
    const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                          / log(grid_size[igrid]/grid_size[igrid-1]);
    if(std::abs(slope_soln_err-poly_degree)>0.05){
        return 1;
    }
    
        pcout << " ********************************************"
             << std::endl
             << " Convergence rates for p = " << poly_degree
             << std::endl
             << " ********************************************"
             << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_Linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("soln_Linf_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

    }//end poly degree loop
}

//}//end PHiLiP namespace
