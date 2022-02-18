#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/convergence_table.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
//#include "numerical_flux/numerical_flux.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "advection_explicit_periodic.h"

#include<fenv.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <fstream>

namespace PHiLiP {
namespace Tests {
template <int dim, int nstate>
AdvectionPeriodic<dim, nstate>::AdvectionPeriodic(const PHiLiP::Parameters::AllParameters *const parameters_input)
:
TestsBase::TestsBase(parameters_input)
, mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}

/****************
 * For Curved Grid Polynomial Representation
 * ************************************/
//dealii::Point<2> CurvManifold::pull_back(const dealii::Point<2> &space_point) const {
template<int dim>
dealii::Point<dim> CurvManifold<dim>::pull_back(const dealii::Point<dim> &space_point) const 
{
    const double pi = atan(1)*4.0;
    dealii::Point<dim> x_ref;
    dealii::Point<dim> x_phys;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = space_point[idim];
        x_phys[idim] = space_point[idim];
    }
    dealii::Vector<double> function(dim);
    dealii::FullMatrix<double> derivative(dim);
    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    //for (int i=0; i<200; i++) {
    int flag =0;
    while(flag != dim){
#if 0
        for(int idim=0;idim<dim;idim++){
            function[idim] = 1.0 / 8.0; 
            //function[idim] = 1.0 / 8.0 /2.0; 
            //function[idim] = 1.0/20.0; 
            for(int idim2=0;idim2<dim;idim2++){
                function[idim] *= std::cos(3.0 * pi/2.0 * x_ref[idim2]);
                //function[idim] *= std::cos(2.0 * pi* x_ref[idim2]);
            }
            function[idim] += x_ref[idim] - x_phys[idim];
        }
#endif
//#if 0
    if(dim==2){
     //   #if 0
        function[0] = x_ref[0] - x_phys[0] +beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function[1] = x_ref[1] - x_phys[1] +beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
     //   #endif
        //nonsym J. Chan
     #if 0
        function[0] =x_ref[0] - x_phys[0] + beta* std::sin(pi*4.0 * (x_ref[0])) * std::sin(pi * (x_ref[1]+1.0));
        function[1] =x_ref[1] - x_phys[1] -  beta*0.5*std::sin(pi*2.0 * (x_ref[0])) * std::sin(pi /2.0 * (x_ref[1]+1.0));
        #endif
        //Gassner
        #if 0
        function[0] =x_ref[0] - x_phys[0] - 0.1*std::sin(2.0*pi*x_ref[1]); 
        function[1] =x_ref[1] - x_phys[1] + 0.1*std::sin(2.0*pi*x_ref[0]);
        #endif
    }
    else if(dim==3){
        #if 0
        function[0] = x_ref[0] - x_phys[0] +beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        function[1] = x_ref[1] - x_phys[1] +beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        function[2] = x_ref[2] - x_phys[2] +beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        #endif
        double temp1 = alpha*(std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]));
        double temp2 = alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]));
        function[0] = x_ref[0] - x_phys[0] + temp1;
        function[1] = x_ref[1] - x_phys[1] + temp2;
        function[2] = x_ref[2] - x_phys[2] +1.0/20.0*( std::sin(2.0 * pi * (temp1+x_ref[0])) + std::sin(2.0 * pi * (temp2+x_ref[1])));
    }
//#endif
    #if 0
        for(int idim=0; idim<dim; idim++){
            for(int idim2=0; idim2<dim;idim2++){
                derivative[idim][idim2] = - 3.0 * pi / 16.0;
                //derivative[idim][idim2] = - 3.0 * pi / 16.0 /2.0;
                //derivative[idim][idim2] = - 1.0/20.0*2.0 * pi;
                for(int idim3 =0;idim3<dim; idim3++){
                    if(idim2 == idim3)
                        derivative[idim][idim2] *=std::sin(3.0 * pi/2.0 * x_ref[idim3]);
                        //derivative[idim][idim2] *=std::sin(2.0 * pi * x_ref[idim3]);
                    else
                        derivative[idim][idim2] *=std::cos(3.0 * pi/2.0 * x_ref[idim3]);
                        //derivative[idim][idim2] *=std::cos(2.0 * pi* x_ref[idim3]);
                }
                if(idim == idim2)
                    derivative[idim][idim2] += 1.0;
            }
        }
#endif
//#if 0
    if(dim==2){
  //  #if 0
        derivative[0][0] = 1.0 - beta* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        derivative[0][1] =  - beta*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        derivative[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        derivative[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
  //  #endif
    //J. Chan nonsym

    #if 0
        derivative[0][0] =1.0 + beta* pi*4.0 * std::cos(pi*4.0 * (x_ref[0])) * std::sin(pi * (x_ref[1]+1.0));
        derivative[0][1] =      beta* pi * std::sin(pi*4.0 * (x_ref[0])) * std::cos(pi * (x_ref[1]+1.0));

        derivative[1][0] =    -  beta*pi *2.0 * 0.5*std::cos(pi*2.0 * (x_ref[0])) * std::sin(pi /2.0 * (x_ref[1]+1.0));
        derivative[1][1] =1.0 -  beta*pi /2.0 * 0.5*std::sin(pi*2.0 * (x_ref[0])) * std::cos(pi /2.0 * (x_ref[1]+1.0));
    #endif
        //Gassner
        #if 0
        derivative[0][0] = 1.0;        
        derivative[0][1] = - 2.0 * pi * 0.1*std::cos(2.0 * pi * x_ref[1]);
                          
        derivative[1][0] =   2.0 * pi * 0.1*std::cos(2.0 * pi * x_ref[0]);
        derivative[1][1] = 1.0;
        #endif
    }
    else if(dim==3){
    #if 0
        derivative[0][0] = 1.0 - beta* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        derivative[0][1] =  - beta*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        derivative[0][2] =  beta*2.0*pi * std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(2.0*pi*(x_ref[2]));

        derivative[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        derivative[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));  
        derivative[1][2] =  beta*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::cos(3.0*pi/2.0*(x_ref[2]));

        derivative[2][0] = beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        derivative[2][1] = - beta*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        derivative[2][2] = 1.0 + beta*5.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(5.0*pi/2.0*(x_ref[2]));
    #endif
        derivative[0][0] = 1.0 + alpha*pi*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
        derivative[0][1] =  alpha*pi*std::sin(pi*x_ref[0])*std::cos(pi*x_ref[1]);
        derivative[0][2] = 0.0; 

        derivative[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
        derivative[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[1])) +alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::cos(pi*x_ref[1]));  
        derivative[1][2] =  0.0;
        double xtemp = x_ref[0] + alpha*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
        double ytemp = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
        derivative[2][0] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*derivative[0][0] + std::cos(2.0*pi*ytemp)*derivative[1][0]);
        derivative[2][1] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*derivative[0][1] + std::cos(2.0*pi*ytemp)*derivative[1][1]);
        derivative[2][2] = 1.0;
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
        function_check[idim] = 1.0/8.0;
        //function_check[idim] = 1.0/8.0 /2.0;
        //function_check[idim] = 1.0/20.0;
        for(int idim2=0; idim2<dim; idim2++){
            function_check[idim] *= std::cos(3.0 * pi/2.0 * x_ref[idim2]);
            //function_check[idim] *= std::cos(2.0 * pi * x_ref[idim2]);
        }
        function_check[idim] += x_ref[idim];
    }
#endif
//#if 0
    if(dim==2){
  //  #if 0
        function_check[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function_check[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
  //  #endif
    //J.Chan nonsym
    #if 0
        function_check[0] =x_ref[0] +beta* std::sin(pi*4.0 * (x_ref[0])) * std::sin(pi * (x_ref[1]+1.0));
        function_check[1] =x_ref[1] -beta* 0.5*std::sin(pi*2.0 * (x_ref[0])) * std::sin(pi /2.0 * (x_ref[1]+1.0));
    #endif
        //Gassner
        #if 0
        function_check[0] =x_ref[0] - 0.1*std::sin(2.0*pi*x_ref[1]); 
        function_check[1] =x_ref[1] + 0.1*std::sin(2.0*pi*x_ref[0]);

        #endif
    }
    else if(dim==3){
    #if 0
        function_check[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        function_check[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        function_check[2] = x_ref[2] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
    #endif
        double temp1 = alpha*(std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]));
        double temp2 = alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]));
        function_check[0] = x_ref[0] + temp1;
        function_check[1] = x_ref[1] + temp2;
        function_check[2] = x_ref[2] +1.0/20.0*( std::sin(2.0 * pi * (temp1+x_ref[0])) + std::sin(2.0 * pi * (temp2+x_ref[1])));
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
        x_phys[idim] = 1.0/8.0;
       // x_phys[idim] = 1.0/8.0 /2.0;
        //x_phys[idim] = 1.0/20.0;
        for(int idim2=0;idim2<dim; idim2++){
           x_phys[idim] *= std::cos( 3.0 * pi/2.0 * x_ref[idim2]);
           //x_phys[idim] *= std::cos( 2.0 * pi * x_ref[idim2]);
        }
        x_phys[idim] += x_ref[idim];
    }
#endif
//#if 0
    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    if(dim==2){
 //   #if 0
        x_phys[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        x_phys[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
  //  #endif
    //J. Chan nonsym
    #if 0
        x_phys[0] =x_ref[0] +beta*  std::sin(pi*4.0 * (x_ref[0])) * std::sin(pi * (x_ref[1]+1.0));
        x_phys[1] =x_ref[1] - beta* 0.5*std::sin(pi*2.0 * (x_ref[0])) * std::sin(pi /2.0 * (x_ref[1]+1.0));
    #endif
        //Gassner
        #if 0
        x_phys[0] =x_ref[0] - 0.1*std::sin(2.0*pi*x_ref[1]); 
        x_phys[1] =x_ref[1] + 0.1*std::sin(2.0*pi*x_ref[0]);
        #endif
    }
    else if(dim==3){
    #if 0
        x_phys[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        x_phys[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        x_phys[2] = x_ref[2] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
    #endif
        double temp1 = alpha*(std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]));
        double temp2 = alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]));
        x_phys[0] =x_ref[0] + temp1; 
        x_phys[1] =x_ref[1] + temp2; 
        x_phys[2] =x_ref[2] +  1.0/20.0*( std::sin(2.0 * pi * (temp1+x_ref[0])) + std::sin(2.0 * pi * (temp2+x_ref[1])));
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
            dphys_dref[idim][idim2] = - 3.0 * pi / 16.0;
            //dphys_dref[idim][idim2] = - 3.0 * pi / 16.0 /2.0;
            //dphys_dref[idim][idim2] = - 1.0/20.0*2.0 * pi;
            for(int idim3 =0;idim3<dim; idim3++){
                if(idim2 == idim3)
                    dphys_dref[idim][idim2] *=std::sin(3.0 * pi/2.0 * x[idim3]);
                    //dphys_dref[idim][idim2] *=std::sin(2.0 * pi * x[idim3]);
                else
                     dphys_dref[idim][idim2] *=std::cos(3.0 * pi/2.0 * x[idim3]);
                    // dphys_dref[idim][idim2] *=std::cos(2.0 * pi* x[idim3]);
            }     
            if(idim == idim2)
                dphys_dref[idim][idim2] += 1.0;
        }
    }
#endif
//#if 0
    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    dealii::Point<dim> x_ref;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = chart_point[idim];
    }

    if(dim==2){
  //  #if 0
        dphys_dref[0][0] = 1.0 - beta*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        dphys_dref[0][1] =  - beta*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        dphys_dref[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        dphys_dref[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
  //  #endif
    //J.Chan nonsym
    #if 0
        dphys_dref[0][0] =1.0 +beta*  pi*4.0 * std::cos(pi*4.0 * (x_ref[0])) * std::sin(pi * (x_ref[1]+1.0));
        dphys_dref[0][1] =      beta* pi * std::sin(pi*4.0 * (x_ref[0])) * std::cos(pi * (x_ref[1]+1.0));

        dphys_dref[1][0] =    - beta*pi *2.0 * 0.5*std::cos(pi*2.0 * (x_ref[0])) * std::sin(pi /2.0 * (x_ref[1]+1.0));
        dphys_dref[1][1] =1.0 - beta*pi /2.0 * 0.5*std::sin(pi*2.0 * (x_ref[0])) * std::cos(pi /2.0 * (x_ref[1]+1.0));
    #endif
        //Gassner
        #if 0
        dphys_dref[0][0] = 1.0;        
        dphys_dref[0][1] = - 2.0 * pi * 0.1*std::cos(2.0 * pi * x_ref[1]);
                
        dphys_dref[1][0] =   2.0 * pi * 0.1*std::cos(2.0 * pi * x_ref[0]);
        dphys_dref[1][1] = 1.0;
        #endif
    }
    else if(dim==3){
    #if 0
        dphys_dref[0][0] = 1.0 - beta*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        dphys_dref[0][1] =  - beta*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        dphys_dref[0][2] =  beta*2.0*pi * std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(2.0*pi*(x_ref[2]));

        dphys_dref[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        dphys_dref[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));  
        dphys_dref[1][2] =  beta*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::cos(3.0*pi/2.0*(x_ref[2]));

        dphys_dref[2][0] = beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        dphys_dref[2][1] = -beta*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        dphys_dref[2][2] = 1.0 + beta*5.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(5.0*pi/2.0*(x_ref[2]));
    #endif
        dphys_dref[0][0] = 1.0 + alpha*pi*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
        dphys_dref[0][1] =  alpha*pi*std::sin(pi*x_ref[0])*std::cos(pi*x_ref[1]);
        dphys_dref[0][2] = 0.0; 

        dphys_dref[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
        dphys_dref[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[1])) +alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::cos(pi*x_ref[1]));  
        dphys_dref[1][2] =  0.0;
        double xtemp = x_ref[0] + alpha*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
        double ytemp = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
        dphys_dref[2][0] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*dphys_dref[0][0] + std::cos(2.0*pi*ytemp)*dphys_dref[1][0]);
        dphys_dref[2][1] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*dphys_dref[0][1] + std::cos(2.0*pi*ytemp)*dphys_dref[1][1]);
        dphys_dref[2][2] = 1.0;
    }
//#endif

    return dphys_dref;
}

template<int dim>
std::unique_ptr<dealii::Manifold<dim,dim> > CurvManifold<dim>::clone() const 
{
    return std::make_unique<CurvManifold<dim>>();
}

template <int dim, int nstate>
dealii::Point<dim> AdvectionPeriodic<dim,nstate>
::warp (const dealii::Point<dim> &p)
{
    const double pi = atan(1)*4.0;
    dealii::Point<dim> q = p;

    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    if (dim == 1){
       // q[dim-1] = p[dim-1] + 1.0/8.0 * cos(3.0 * pi/2.0 * p[dim-1]);
        q[dim-1] = p[dim-1] + cos(2.0 * pi* p[dim-1]);
    }
    if (dim == 2){
#if 0
        q[dim-1] = p[dim-1] + 1.0/8.0 * std::cos(3.0 * pi/2.0 * p[dim-1]) * std::cos(3.0 * pi/2.0 * p[dim-2]);
        q[dim-2] = p[dim-2] + 1.0/8.0 * std::cos(3.0 * pi/2.0 * p[dim-1]) * std::cos(3.0 * pi/2.0 * p[dim-2]);
#endif
        //non sym transform
    //    #if 0
        q[dim-2] =p[dim-2] +  beta*std::cos(pi/2.0 * p[dim-2]) * std::cos(3.0 * pi/2.0 * p[dim-1]);
        q[dim-1] =p[dim-1] +  beta*std::sin(2.0 * pi * (p[dim-2])) * std::cos(pi /2.0 * p[dim-1]);
    //    #endif
        //Jesse Chan non sym transform
#if 0
        q[0] =p[0] +beta* std::sin(pi*4.0 * (p[0])) * std::sin(pi * (p[1]+1));
        q[1] =p[1] -beta* 0.5*std::sin(pi*2.0 * (p[0])) * std::sin(pi /2.0 * (p[1]+1));
#endif
        //Gassner nonsym
        #if 0
        q[0] =p[0] - 0.1*std::sin(2.0*pi*p[1]); 
        q[1] =p[1] + 0.1*std::sin(2.0*pi*p[0]);
        #endif

    }
    if(dim==3){
       //q[dim-1] = p[dim-1] + 1.0/16.0 * cos(3.0 * pi/2.0 * p[dim-1]) * cos(3.0 * pi/2.0 * p[dim-2]) * cos(3.0 * pi/2.0 * p[dim-3]);
       //q[dim-2] = p[dim-2] + 1.0/16.0 * cos(3.0 * pi/2.0 * p[dim-1]) * cos(3.0 * pi/2.0 * p[dim-2]) * cos(3.0 * pi/2.0 * p[dim-3]);
       //q[dim-3] = p[dim-3] + 1.0/16.0 * cos(3.0 * pi/2.0 * p[dim-1]) * cos(3.0 * pi/2.0 * p[dim-2]) * cos(3.0 * pi/2.0 * p[dim-3]);
       //transform David
       #if 0
        q[dim-1] =p[dim-1] + 1.0/20.0*  std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi * p[dim-2]) * std::cos(2.0 * pi * p[dim-3]);
        q[dim-2] =p[dim-2] +  1.0/20.0* std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi * p[dim-2]) * std::cos(2.0 * pi * p[dim-3]);
        q[dim-3] =p[dim-3] +  1.0/20.0* std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi * p[dim-2]) * std::cos(2.0 * pi * p[dim-3]);
        #endif
        //non sym transform
        #if 0
        q[0] =p[0] +  beta*std::cos(pi/2.0 * p[0]) * std::cos(3.0 * pi/2.0 * p[1]) * std::sin(2.0 * pi * (p[2]));
        q[1] =p[1] +  beta*std::sin(2.0 * pi * (p[0])) * std::cos(pi /2.0 * p[1]) * std::sin(3.0 * pi /2.0 * p[2]);
        q[2] =p[2] +  beta*std::sin(2.0 * pi * (p[0])) * std::cos(3.0 * pi/2.0 * p[1]) * std::cos(5.0 * pi/2.0 * p[2]);
        #endif
       //periodic non sym wild
        double temp1 = alpha*(std::sin(pi * p[0]) * std::sin(pi * p[1]));
        double temp2 = alpha*exp(1.0-p[1])*(std::sin(pi * p[0]) * std::sin(pi* p[1]));
        q[0] =p[0] + temp1; 
        q[1] =p[1] + temp2;
        q[2] =p[2] +  1.0/20.0*( std::sin(2.0 * pi * q[0]) + std::sin(2.0 * pi * q[1]));

    }

    return q;
}
/****************************
 * End of Curvilinear Grid
 * ***************************/

template<int dim, int nstate>
double AdvectionPeriodic<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const
{
	double energy = 0.0;
       // dealii::LinearAlgebra::distributed::Vector<double> Mu_hat(dg->solution.size());
        dealii::LinearAlgebra::distributed::Vector<double> Mu_hat(dg->right_hand_side);
       // printf(" size Mass %d size sol %d\n",dg->global_mass_matrix.n(),dg->solution.size());
       // printf("  Mass %g size sol %g\n",dg->global_mass_matrix(dg->global_mass_matrix.m(),dg->global_mass_matrix.n()),dg->solution(dg->solution.size()));
        dg->global_mass_matrix.vmult( Mu_hat, dg->solution);
        energy = dg->solution * Mu_hat;
#if 0
      //  printf("didn't fail where thought\n");
//	for (unsigned int i = 0; i < dg->solution.size(); ++i)
	for (unsigned int i = 0; i < dg->right_hand_side.size(); ++i)
	{
	//	energy += dg->global_mass_matrix(i,i) * dg->solution(i) * dg->solution(i);
            energy +=  dg->solution(i) * Mu_hat(i);
	}
#endif
//	return energy;
	return sqrt(energy);
}

template<int dim, int nstate>
double AdvectionPeriodic<dim, nstate>::compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const
{
	double conservation = 0.0;
       // dealii::LinearAlgebra::distributed::Vector<double> Mu_hat(dg->solution.size());
        dealii::LinearAlgebra::distributed::Vector<double> Mu_hat(dg->right_hand_side);
       // printf(" size Mass %d size sol %d\n",dg->global_mass_matrix.n(),dg->solution.size());
       // printf("  Mass %g size sol %g\n",dg->global_mass_matrix(dg->global_mass_matrix.m(),dg->global_mass_matrix.n()),dg->solution(dg->solution.size()));
        //dg->global_mass_matrix.vmult( Mu_hat, dg->right_hand_side);
        dg->global_mass_matrix.vmult( Mu_hat, dg->solution);

        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        dealii::FullMatrix<double> Chi_operator(n_quad_pts, n_dofs_cell);
        for(int istate=0; istate<nstate; istate++){
            for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[poly_degree].point(iquad);
                    Chi_operator[iquad][itest] = dg->fe_collection[poly_degree].shape_value_component(itest,qpoint,istate);
                }
            }
        }
        dealii::FullMatrix<double> Chi_inv(n_dofs_cell);
        Chi_inv.invert(Chi_operator);
        dealii::Vector<double> ones(n_quad_pts);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            ones[iquad] = 1.0;
        }
        dealii::Vector<double> ones_hat(n_dofs_cell);
        Chi_inv.vmult(ones_hat, ones);

        dealii::LinearAlgebra::distributed::Vector<double> ones_hat_global(dg->right_hand_side);
#if 0
        const unsigned int number_cells = dg->right_hand_side.size() / n_dofs_cell;
        for( unsigned int cell = 0; cell<number_cells; cell++){
            for(unsigned int i= cell * n_dofs_cell, j=0; i< (cell+1) * n_dofs_cell && j<n_dofs_cell; i++, j++){
                ones_hat_global[i] = ones_hat[j];
            }
        }
#endif
        std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
        for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices (dofs_indices);
            for(unsigned int idof=0;idof<n_dofs_cell; idof++){
                ones_hat_global[dofs_indices[idof]] = ones_hat[idof];
            }
        }

        conservation = ones_hat_global * Mu_hat;

	return conservation;
}

template <int dim, int nstate>
int AdvectionPeriodic<dim, nstate>::run_test() const
{
#if 0
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
			dealii::Triangulation<dim> grid(
				typename dealii::Triangulation<dim>::MeshSmoothing(
					dealii::Triangulation<dim>::smoothing_on_refinement |
					dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
			dealii::parallel::distributed::Triangulation<dim> grid(
				this->mpi_communicator);
#endif
#endif

printf("starting test\n");
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  

    const unsigned int n_grids = (all_parameters_new.use_energy) ? 4 : 5;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);
    std::vector<double> soln_error_inf(n_grids);
   // std::array<double,n_grids> output_error;
    using ADtype = double;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

#if 0
    std::vector<int> n_1d_cells(n_grids);
    n_1d_cells[0] =1;
    for (unsigned int igrid=1;igrid<n_grids;++igrid) {
        n_1d_cells[igrid] = static_cast<int>(n_1d_cells[igrid-1]*1.5) + 2;
    }
#endif
	//double left = 0.0;
//	double right = 2.0;
	const double left = -1.0;
	//const double left = 0.0;
	const double right = 1.0;
	//double left = -5.0;
	//double right = 5.0;
	//double left = -1.0;
	//double right = 1.0;
	//double right = 2.0;
	const bool colorize = true;
//	int n_refinements = 6;
	unsigned int n_refinements = n_grids;
	unsigned int poly_degree = 3;
        unsigned int grid_degree = poly_degree;
//	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);

        dealii::ConvergenceTable convergence_table;
        const unsigned int igrid_start = (all_parameters_new.use_energy) ? 3 : 3;
printf("NEW GRID\n");
fflush(stdout);

    for(unsigned int igrid=igrid_start; igrid<n_refinements; ++igrid){
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
//#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
//			dealii::Triangulation<dim> grid(
//				typename dealii::Triangulation<dim>::MeshSmoothing(
//					dealii::Triangulation<dim>::smoothing_on_refinement |
//					dealii::Triangulation<dim>::smoothing_on_coarsening));
//#else
//			dealii::parallel::distributed::Triangulation<dim> grid(
//				this->mpi_communicator);
//       // dealii::parallel::distributed::Triangulation<dim> grid(
//        //    this->mpi_communicator,
//         //   typename dealii::Triangulation<dim>::MeshSmoothing(
//          //      dealii::Triangulation<dim>::smoothing_on_refinement |
//           //     dealii::Triangulation<dim>::smoothing_on_coarsening));
//#endif


//testing warping
dealii::GridGenerator::hyper_cube (*grid, left, right, colorize);


 #if PHILIP_DIM==1
#else
	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
                if(dim == 2)
		dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
                if(dim==3)
		dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
		grid->add_periodicity(matched_pairs);
#endif

	grid->refine_global(igrid);

//warp curvilinear transformed grid
//#if 0
dealii::GridTools::transform (&warp, *grid);

// Assign a manifold to have curved geometry
        const CurvManifold<dim> curv_manifold;
        unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
        grid->reset_all_manifolds();
        grid->set_all_manifold_ids(manifold_id);
        grid->set_manifold ( manifold_id, curv_manifold );
//#endif
//end of warp curv transformed grid

#if 0
            const unsigned int n_global_active_cells2 = grid.n_global_active_cells();
    if(poly_degree == 2)
    all_parameters_new.ode_solver_param.initial_time_step = 0.512*2.0/n_global_active_cells2 ;
    else
    all_parameters_new.ode_solver_param.initial_time_step = 0.128*2.0/n_global_active_cells2 ;
#endif
//CFL number
    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double n_dofs_cfl = pow(n_global_active_cells2,dim) * pow(poly_degree+1.0, dim);
    double delta_x = (right-left)/pow(n_dofs_cfl,(1.0/dim)); 
   // all_parameters_new.ode_solver_param.initial_time_step =  delta_x /(2.0*(2.0*poly_degree+1)) ;
    //all_parameters_new.ode_solver_param.initial_time_step =  delta_x /(1.0*(2.0*poly_degree+1)) ;
    all_parameters_new.ode_solver_param.initial_time_step =  delta_x /(1.0*(2.0*poly_degree+1)) ;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.25*delta_x;
    all_parameters_new.ode_solver_param.initial_time_step =  0.5*delta_x;
    all_parameters_new.ode_solver_param.initial_time_step =  (all_parameters_new.use_energy) ? 0.05*delta_x : 0.5*delta_x;
  //  all_parameters_new.ode_solver_param.initial_time_step =  0.001;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.8*delta_x;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.25*delta_x;
    //all_parameters_new.ode_solver_param.initial_time_step =  0.01;
   // all_parameters_new.ode_solver_param.initial_time_step =  1.0*delta_x;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.05*delta_x;
    //all_parameters_new.ode_solver_param.initial_time_step =  0.05*delta_x/3.0;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.005*delta_x;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.0005*delta_x;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.000000005*delta_x;
    std::cout << "dt " <<all_parameters_new.ode_solver_param.initial_time_step <<  std::endl;
    std::cout << "cells " <<n_global_active_cells2 <<  std::endl;
  //  all_parameters_new.ode_solver_param.initial_time_step *= 1e-3;  

//	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, &grid);
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
	dg->allocate_system ();

//	for (auto current_cell = dg->dof_handler.begin_active(); current_cell != dg->dof_handler.end(); ++current_cell) {
//		 if (!current_cell->is_locally_owned()) continue;
//
//		 dg->fe_values_volume.reinit(current_cell);
//		 int cell_index = current_cell->index();
//		 std::cout << "cell number " << cell_index << std::endl;
//		 for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<PHILIP_DIM>::faces_per_cell; ++face_no)
//		 {
//			 if (current_cell->face(face_no)->at_boundary())
//		     {
//				 std::cout << "face " << face_no << " is at boundary" << std::endl;
//		         typename dealii::DoFHandler<PHILIP_DIM>::active_cell_iterator neighbor = current_cell->neighbor_or_periodic_neighbor(face_no);
//		         std::cout << "the neighbor is " << neighbor->index() << std::endl;
//		     }
//		 }
//
//	}

	std::cout << "Implement initial conditions" << std::endl;
#if 0
	dealii::FunctionParser<dim> initial_condition;
	std::string variables;
        if (dim == 3)
	    variables = "x,y,z";
        if (dim == 2)
	    variables = "x,y";
        if (dim == 1)
	    variables = "x";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::string expression;
        if (dim == 3)
	    expression = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) + 20*(z-1)*(z-1) ) )";//"sin(pi*x)*sin(pi*y)";
        if (dim == 2){
	    //expression = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) ) )";//"sin(pi*x)*sin(pi*y)";
	    //expression = "exp( -( 20*(x-2)*(x-2) + 20*(y-2)*(y-2) ) )";//"sin(pi*x)*sin(pi*y)";
        if (all_parameters_new.use_energy == true){//for split form get energy
	    expression = "exp( -( 20*(x)*(x) + 20*(y)*(y) ) )";//"sin(pi*x)*sin(pi*y)";
	    //expression = "exp( -( (x)*(x) + (y)*(y) ) )";//"sin(pi*x)*sin(pi*y)";
            }else{
	    expression = "sin(pi*x)*sin(pi*y)";//"sin(pi*x)*sin(pi*y)";
	    expression = "sin(pi*(x+y))";//"sin(pi*x)*sin(pi*y)";
            }}
	    //expression = "exp( -( 20*(x-1.5)*(x-1.5) + 20*(y-1.5)*(y-1.5) ) )";//"sin(pi*x)*sin(pi*y)";
        if(dim==1)
	    expression = "exp( -( 20*(x)*(x) ) )";//"sin(pi*x)*sin(pi*y)";
	    //expression = "sin(pi*x)";//"sin(pi*x)*sin(pi*y)";
	initial_condition.initialize(variables,
								 expression,
								 constants);
	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
	
#endif
	
//#if 0
//if use curvilinear grid to interpolate IC
                const unsigned int n_quad_pts1      = dg->volume_quadrature_collection[poly_degree].size();
                const unsigned int n_dofs_cell1     =dg->fe_collection[poly_degree].dofs_per_cell;
         //   dealii::QGauss<dim> quad_val(poly_degree+1);
            //dealii::QGaussLobatto<dim> quad_val(poly_degree+1);
           // const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid.mapping_fe_field));
           // const dealii::MappingQGeneric<dim, dim> mapping(poly_degree+1);
            dealii::FEValues<dim,dim> fe_values_test(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], dg->volume_quadrature_collection[poly_degree], 
           // dealii::FEValues<dim,dim> fe_values_test(mapping, dg->fe_collection[poly_degree], dg->volume_quadrature_collection[poly_degree], 
            //        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
                                dealii::update_values | dealii::update_JxW_values | 
                                dealii::update_jacobians |  
                                dealii::update_quadrature_points | dealii::update_inverse_jacobians);
           // const unsigned int n_quad_pts1 = fe_values_test.n_quadrature_points;
           #if 0
        dealii::FullMatrix<double> Chi_operator(n_quad_pts1, n_dofs_cell1);
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell1; ++itest) {
        for (unsigned int iquad=0; iquad<n_quad_pts1; ++iquad) {
            const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[poly_degree].point(iquad);
            Chi_operator[iquad][itest] = dg->fe_collection[poly_degree].shape_value_component(itest,qpoint,istate);
           // Chi_operator[iquad][itest] = fe_values_test.shape_value_component(itest,iquad,istate);
        }
    }
    }
    dealii::FullMatrix<double> Chi_inv_operator(n_quad_pts1, n_dofs_cell1);
    Chi_inv_operator.invert(Chi_operator);
#endif
            const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
            const double pi2 = atan(1)*4.0;
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
	
                fe_values_test.reinit (current_cell);
                current_dofs_indices.resize(n_dofs_cell1);
                current_cell->get_dof_indices (current_dofs_indices);
                for(unsigned int idof=0; idof<n_dofs_cell1; idof++){
                    dg->solution[current_dofs_indices[idof]]=0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts1; iquad++){
                    //const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[poly_degree].point(iquad);
                    const dealii::Point<dim> qpoint = (fe_values_test.quadrature_point(iquad));
                    double exact = 1.0;
                       for (int idim=0; idim<dim; idim++){
                    //   for (int idim=0; idim<1; idim++){
                            if (all_parameters_new.use_energy == true){//for split form get energy
                                if(left == 0){
                                    exact *= exp(-20*(qpoint[idim]-0.5)*(qpoint[idim]-0.5));
                                }
                                if(left == -1){
                                    exact *= exp(-20*(qpoint[idim])*(qpoint[idim]));
                                }
                                //exact *= exp(-(qpoint[idim])*(qpoint[idim]));
                            }
                            else{
                                if(left == 0){
                                    exact *= sin(2.0*pi2*qpoint[idim]);
                                }
                                if(left==-1){
                                    exact *= sin(pi2*qpoint[idim]);
                                }
                               // exact *= sin(0.5*pi2*qpoint[idim]);
                                //exact *= exp(-(qpoint[idim])*(qpoint[idim]));
                                //exact *= exp(-20.0*(qpoint[idim]+0.5)*(qpoint[idim]+0.5));
                            }
                       // exact = sin(2.0*pi2*((qpoint[0]) + (qpoint[dim-1])));//for grid 1-3
                        }
#if 0
                    double exact2 = 1.0;
                       for (int idim=0; idim<dim; idim++){
                                exact2 *= sin(2.0*pi2*qpoint[idim]);
                        }
                    double exact3 = 1.0;
                       for (int idim=0; idim<dim; idim++){
                                exact3 *= sin(4.0*pi2*qpoint[idim]);
                        }
                    double exact4 = 1.0;
                       for (int idim=0; idim<dim; idim++){
                                exact4 *= sin(6.0*pi2*qpoint[idim]);
                        }
                        exact += exact2+exact3+exact4;
#endif
                       // dg->solution[current_dofs_indices[idof]] +=Chi_inv_operator[idof][iquad] *exact; 
                        dg->solution[current_dofs_indices[idof]] +=dg->operators.vol_projection_operator[poly_degree][idof][iquad] *exact; 
                    }   
                }


}
//#endif
	
//Get new dx as min dist between 2 quad points in physical space

#if 0
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
                    fe_values_test.reinit (current_cell);
                    for(unsigned int iquad=1; iquad<n_quad_pts1; iquad++){
                        const dealii::Point<dim> qpoint = (fe_values_test.quadrature_point(iquad));
                        const dealii::Point<dim> qpoint_past = (fe_values_test.quadrature_point(iquad-1));
                        for(int idim=0; idim<dim; idim++){
                            if(std::abs(qpoint[idim] - qpoint_past[idim])<delta_x){
                                delta_x = std::abs(qpoint[idim] - qpoint_past[idim]);
                            } 
                        }
                    }
            }
            all_parameters_new.ode_solver_param.initial_time_step =  0.5*delta_x;

#endif



	// Create ODE solver using the factory and providing the DG object
	std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

	double finalTime = 1.5;
finalTime = 10.0;
//finalTime = 5.0;
finalTime = 2.0;
//finalTime=0.25;
//finalTime =1/8;
//finalTime = 0.5;
//finalTime = 10.0;
//finalTime = 1.0;
//finalTime = all_parameters_new.ode_solver_param.initial_time_step;
//finalTime = 10.0*all_parameters_new.ode_solver_param.initial_time_step;

//#if 0
	//need to call ode_solver before calculating energy because mass matrix isn't allocated yet.

        if (all_parameters_new.use_energy == true){//for split form get energy
//	ode_solver->advance_solution_time(0.000001);
	double dt = all_parameters_new.ode_solver_param.initial_time_step;

        ode_solver->current_iteration = 0;

	ode_solver->advance_solution_time(dt/10.0);
	double initial_energy = compute_energy(dg);
	double initial_conservation = compute_conservation(dg, poly_degree);
//printf("initial energy %g\n",initial_energy);

	std::ofstream myfile ("energy_plot.gpl" , std::ios::trunc);

        ode_solver->current_iteration = 0;

	for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
	{
		ode_solver->advance_solution_time(dt);
        //    #if 0
		double current_energy = compute_energy(dg);
                current_energy /=initial_energy;
                std::cout << std::setprecision(16) << std::fixed;
		pcout << "Energy at time " << i * dt << " is " << current_energy<< std::endl;
		myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_energy << std::endl;
		if (current_energy*initial_energy - initial_energy >= 1.00)//since normalized by initial
		//if (current_energy - initial_energy >= 10.00)
		{
                    pcout << " Energy was not monotonically decreasing" << std::endl;
			return 1;
			break;
		}
		if ( (current_energy*initial_energy - initial_energy >= 1.0e-12) && (all_parameters_new.conv_num_flux_type == Parameters::AllParameters::ConvectiveNumericalFlux::central_flux))
		{
                    pcout << " Energy was not conserved" << std::endl;
			return 1;
			break;
		}
        //    #endif
                //Conservation

//		ode_solver->advance_solution_time(dt);
		double current_conservation = compute_conservation(dg, poly_degree);
                current_conservation /=initial_conservation;
                std::cout << std::setprecision(16) << std::fixed;
		pcout << "Normalized Conservation at time " << i * dt << " is " << current_conservation<< std::endl;
		myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_conservation << std::endl;
		if (current_conservation*initial_conservation - initial_conservation >= 10.00)
		//if (current_energy - initial_energy >= 10.00)
		{
                    pcout << "Not conserved" << std::endl;
			return 1;
			break;
		}

	}
	myfile.close();
        }
        else{//do OOA
            if(left==-1){
               // finalTime = 2.0;
                finalTime = 0.5;
            }
            if(left==0){
                finalTime=1.0;
            }
            //finalTime = 1.0;
          //  finalTime = 0.5;
          //  finalTime = 0.1;
           // finalTime = 2.5;
            //finalTime = 0.5;
          //  finalTime = all_parameters_new.ode_solver_param.initial_time_step;
          //  finalTime = 2.0*all_parameters_new.ode_solver_param.initial_time_step;
          //  finalTime = 10.0*all_parameters_new.ode_solver_param.initial_time_step;
           // finalTime = 0.0;

pcout<<" dim "<<dim<<std::endl;
        ode_solver->current_iteration = 0;

	    ode_solver->advance_solution_time(finalTime);

//#endif

	//ode_solver->advance_solution_time(finalTime);

//output results
            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim
                 << "\t Polynomial degree p: " << poly_degree
                 << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
           // overintegrate = 0;
            //dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            //dealii::QGaussLobatto<dim> quad_extra(poly_degree+1+overintegrate);
          //  dealii::MappingQ<dim,dim> mappingq(poly_degree+1+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
           // dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                 //              dealii::update_values | dealii::update_JxW_values | 
                 //              dealii::update_jacobians |  
                 //              dealii::update_quadrature_points | dealii::update_inverse_jacobians);
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0.0;

            // Integrate solution error and output error

            const double pi = atan(1)*4.0;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            double linf_error = 0.0;
            const dealii::Tensor<1,3,double> adv_speeds = Parameters::ManufacturedSolutionParam::get_default_advection_vector();
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    for (int istate=0; istate<nstate; ++istate) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact=1.0;
                    for(int idim=0; idim<dim; idim++){
                //    for(int idim=0; idim<1; idim++){
                        //uexact *= exp(-(20 * (qpoint[idim] - 2) * (qpoint[idim] - 2)));//for grid 1-3
                       // uexact *= exp(-((qpoint[idim]-finalTime) * (qpoint[idim]-finalTime)));//for grid 1-3
            //           if(idim==0)
                        if(left==0){
                            uexact *= sin(2.0*pi*(qpoint[idim]-finalTime));//for grid 1-3
                        }
                        if(left==-1){
                           // uexact *= sin(pi*(qpoint[idim]-finalTime));//for grid 1-3
                            uexact *= sin(pi*(qpoint[idim]- adv_speeds[idim]*finalTime));//for grid 1-3
                        }
             //           else
              //          uexact *= sin(pi*(qpoint[idim]-0.5*finalTime));//for grid 1-3
                       // uexact *= sin(0.5*pi*(qpoint[idim]-finalTime));//for grid 1-3
                       // uexact *= exp(-20.0*(qpoint[idim]+0.5-finalTime)*(qpoint[idim]+0.5-finalTime));//for grid 1-3
                    }
#if 0
                    double uexact2=1.0;
                    for(int idim=0; idim<dim; idim++){
                        uexact2 *= sin(2.0*pi*(qpoint[idim]-finalTime));//for grid 1-3
                    }
                    double uexact3=1.0;
                    for(int idim=0; idim<dim; idim++){
                        uexact3 *= sin(4.0*pi*(qpoint[idim]-finalTime));//for grid 1-3
                    }
                    double uexact4=1.0;
                    for(int idim=0; idim<dim; idim++){
                        uexact4 *= sin(6.0*pi*(qpoint[idim]-finalTime));//for grid 1-3
                    }
                    uexact += uexact2+uexact3+uexact4;
#endif
                     //   uexact = sin(2.0*pi*((qpoint[0]-finalTime) + (qpoint[dim-1]-finalTime)));//for grid 1-3
                        //std::cout << uexact - soln_at_q[istate] << std::endl;
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                
                        double inf_temp = std::abs(soln_at_q[istate]-uexact);
                            if(inf_temp > linf_error){
                                linf_error = inf_temp;
                            }
            //            l2error += pow(soln_at_q[istate] - uexact, 2);
                    }
                }

            }
           // l2error /= (n_global_active_cells *n_dofs); 
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
            const double linferror_mpi= (dealii::Utilities::MPI::max(linf_error, mpi_communicator));

           // double solution_integral = integrate_solution_over_domain(*dg);

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
            soln_error_inf[igrid] = linferror_mpi;
     //       output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

//#if 0
            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("soln_Linf_error", linferror_mpi);
 //           convergence_table.add_value("output_error", output_error[igrid]);
//#endif


            pcout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                 << " Linf-soln_error: " << linferror_mpi
                 << " Residual: " << ode_solver->residual_norm
                 << std::endl;

#if 0
            pcout //<< " output_exact: " << exact_solution_integral
                 //<< " output_discrete: " << solution_integral
                 << " output_error: " << output_error[igrid]
                 << std::endl;
#endif

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                const double slope_soln_err_inf = log(soln_error_inf[igrid]/soln_error_inf[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
              //  const double slope_output_err = log(output_error[igrid]/output_error[igrid-1])
     //                                 / log(grid_size[igrid]/grid_size[igrid-1]);
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
            
                if(igrid == n_grids-1){
                    if(std::abs(slope_soln_err_inf-(poly_degree+1))>0.1)
                        return 1;
                }            

            }

    }
        pcout << " ********************************************"
             << std::endl
             << " Convergence rates for p = " << poly_degree
             << std::endl
             << " ********************************************"
             << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_Linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("soln_Linf_error", true);
        convergence_table.set_scientific("output_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
        }//end of OOA

       // convergence_table_vector.push_back(convergence_table);

	return 0; //need to change
}

//int main (int argc, char * argv[])
//{
//	//parse parameters first
//	feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
//	dealii::deallog.depth_console(99);
//		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
//		const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
//		const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
//		dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
//		pcout << "Starting program with " << n_mpi << " processors..." << std::endl;
//		if ((PHILIP_DIM==1) && !(n_mpi==1)) {
//			std::cout << "********************************************************" << std::endl;
//			std::cout << "Can't use mpirun -np X, where X>1, for 1D." << std::endl
//					  << "Currently using " << n_mpi << " processors." << std::endl
//					  << "Aborting..." << std::endl;
//			std::cout << "********************************************************" << std::endl;
//			std::abort();
//		}
//	int test_error = 1;
//	try
//	{
//		// Declare possible inputs
//		dealii::ParameterHandler parameter_handler;
//		PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
//		PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);
//
//		// Read inputs from parameter file and set those values in AllParameters object
//		PHiLiP::Parameters::AllParameters all_parameters;
//		std::cout << "Reading input..." << std::endl;
//		all_parameters.parse_parameters (parameter_handler);
//
//		AssertDimension(all_parameters.dimension, PHILIP_DIM);
//
//		std::cout << "Starting program..." << std::endl;
//
//		using namespace PHiLiP;
//		//const Parameters::AllParameters parameters_input;
//		AdvectionPeriodic<PHILIP_DIM, 1> advection_test(&all_parameters);
//		int i = advection_test.run_test();
//		return i;
//	}
//	catch (std::exception &exc)
//	{
//		std::cerr << std::endl << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Exception on processing: " << std::endl
//				  << exc.what() << std::endl
//				  << "Aborting!" << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl;
//		return 1;
//	}
//
//	catch (...)
//	{
//		std::cerr << std::endl
//				  << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Unknown exception!" << std::endl
//				  << "Aborting!" << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl;
//		return 1;
//	}
//	std::cout << "End of program." << std::endl;
//	return test_error;
//}

    template class AdvectionPeriodic <PHILIP_DIM,1>;

} //Tests namespace
} //PHiLiP namespace
