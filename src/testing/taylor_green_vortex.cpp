#include "taylor_green_vortex.h"

namespace PHiLiP {
namespace Tests {

// done
template <int dim, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,real>
::InitialConditionFunction_TaylorGreenVortex (
    const unsigned int nstate,
    const double       gamma_gas,
    const double       mach_inf_sqr)
    : InitialConditionFunction_FlowSolver<dim,real>(nstate)
    , gamma_gas(gamma_gas)
    , mach_inf_sqr(mach_inf_sqr)    
{
    // casting `nstate` as `int` to avoid errors
    static_assert(((int)nstate)==dim+2, "Tests::InitialConditionFunction_TaylorGreenVortex() should be created with nstate=dim+2");
}
// done
template <int dim, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0], y = point[1], z = point[2];
        
        if(istate==0) {
            // density
            value = 1.0;
        }
        if(istate==1) {
            // x-velocity
            value = sin(x)*cos(y)*cos(z); 
        }
        if(istate==2) {  
            // y-velocity
            value = -cos(x)*sin(y)*cos(z);
        }
        if(istate==3) {
            // z-velocity
            value = 0.0;
        }
        if(istate==4) {
            // pressure
            value = 1.0/(gamma_gas*mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(cos(2.0*z)+2.0);
        }
    }
    return value;
}
// done
template <int dim, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if (dim == 3) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real w   = primitive_value(point,3);
        const real p   = primitive_value(point,4);

        // convert primitive to conservative solution
        if(istate==0) value = rho; // density
        if(istate==1) value = rho*u; // x-momentum
        if(istate==2) value = rho*v; // y-momentum
        if(istate==3) value = rho*w; // z-momentum
        if(istate==4) value = p/(1.4-1.0) + 0.5*rho*(u*u + v*v + w*w); // total energy
    }

    return value;
}
// done
template <int dim, typename real>
inline real InitialConditionFunction_TaylorGreenVortex<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}
// done
template <int dim, typename real>
dealii::Tensor<1,dim,real> InitialConditionFunction_TaylorGreenVortex<dim,real>
::primitive_gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    // Gradients of primitive variables 
    if (dim == 2) {
        const real x = point[0], y = point[1], z = point[2];

        if(istate==0) {
            // density
            gradient[0] = 0.0; // dx
            gradient[1] = 0.0; // dy
            gradient[2] = 0.0; // dz
        }
        if(istate==1) {
            // x-velocity
            gradient[0] =  cos(x)*cos(y)*cos(z); // dx
            gradient[1] = -sin(x)*sin(y)*cos(z); // dy
            gradient[2] = -sin(x)*cos(y)*sin(z); // dz
        }
        if(istate==2) {
            // y-velocity
            gradient[0] = -sin(x)*sin(y)*cos(z); // dx
            gradient[1] = -cos(x)*cos(y)*cos(z); // dy
            gradient[2] =  cos(x)*sin(y)*sin(z); // dz
        }
        if(istate==3) {
            // z-velocity
            gradient[0] = 0.0; // dx
            gradient[1] = 0.0; // dy
            gradient[2] = 0.0; // dz
        }
        if(istate==4) {
            // pressure
            gradient[0] = -(1.0/8.0)*sin(2.0*x)*(cos(2.0*z)+2.0); // dx
            gradient[1] = -(1.0/8.0)*sin(2.0*y)*(cos(2.0*z)+2.0); // dy
            gradient[2] = -(1.0/8.0)*(cos(2.0*x)+cos(2.0*y))*sin(2.0*z); // dz
        }
    }
    return gradient;
}
// done
template <int dim, typename real>
dealii::Tensor<1,dim,real> InitialConditionFunction_TaylorGreenVortex<dim,real>
::convert_primitive_to_conversative_gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    if (dim == 3) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real w   = primitive_value(point,3);
        const real p   = primitive_value(point,4);
        const dealii::Tensor<1,dim,real> rho_grad = primitive_gradient(point,0);
        const dealii::Tensor<1,dim,real> u_grad   = primitive_gradient(point,1);
        const dealii::Tensor<1,dim,real> v_grad   = primitive_gradient(point,2);
        const dealii::Tensor<1,dim,real> w_grad   = primitive_gradient(point,3);
        const dealii::Tensor<1,dim,real> p_grad   = primitive_gradient(point,4);
        
        // convert to primitive to gradient of conservative variables using product rule
        if(istate==0) {
            // density
            for(int d=0; d<dim; d++) { 
                gradient[d] = rho_grad[d];
            }
        }
        if(istate==1) {
            // x-momentum
            for(int d=0; d<dim; d++) {
                gradient[d] = u*rho_grad[d] + rho*u_grad[d];
            }
        }
        if(istate==2) {
            // y-momentum
            for(int d=0; d<dim; d++) {
                gradient[d] = v*rho_grad[d] + rho*v_grad[d];
            }
        }
        if(istate==3) {
            // z-momentum
            for(int d=0; d<dim; d++) {
                gradient[d] = w*rho_grad[d] + rho*w_grad[d];
            }
        }
        if(istate==4) {
            // total energy
            for(int d=0; d<dim; d++) {
                gradient[d] = p_grad[d]/(1.4-1.0) + 0.5*rho_grad[d]*(u*u + v*v + w*w) + rho*(u*u_grad[d]+v*v_grad[d]+w*w_grad[d]);
            }
        }
    }
    return gradient;
}
// done
template <int dim, typename real>
inline dealii::Tensor<1,dim,real> InitialConditionFunction_TaylorGreenVortex<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient = convert_primitive_to_conversative_gradient(point, istate);
    return gradient;
}


template <int dim, int nstate>
int TaylorGreenVortex<dim, nstate>::run_test() const
{
    // pcout << " Running Burgers energy stability. " << std::endl;

    // // Grid Generation
    // using Triangulation = dealii::Triangulation<dim>;
    // std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();
   
    // double left = 0.0;
    // double right = 2.0;
    // const bool colorize = true;
    // int n_refinements = 5;
    // unsigned int poly_degree = 7;
    // dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
   
    // std::vector<dealii::GridTools::PeriodicFacePair<typename Triangulation::cell_iterator> > matched_pairs;
    // dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
    // grid->add_periodicity(matched_pairs);
   
   
    // grid->refine_global(n_refinements);
    // pcout << "Grid generated and refined" << std::endl;
}


#if PHILIP_DIM==3
    // InitialConditionFunction
    template class InitialConditionFunction_TaylorGreenVortex <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

