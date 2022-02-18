#include <CoDiPack/include/codi.hpp>
#include <Sacado.hpp>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>
#include <deal.II/base/function_time.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>

#include "manufactured_solution.h"

template class dealii::FunctionTime<Sacado::Fad::DFad<double>>; // Needed by Function
template class dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>;

namespace PHiLiP {

///< Provide isfinite for double.
bool isfinite(double value)
{
    return std::isfinite(static_cast<double>(value));
}

///< Provide isfinite for FadType
bool isfinite(Sacado::Fad::DFad<double> value)
{
    return std::isfinite(static_cast<double>(value.val()));
}

///< Provide isfinite for FadFadType
bool isfinite(Sacado::Fad::DFad<Sacado::Fad::DFad<double>> value)
{
    return std::isfinite(static_cast<double>(value.val().val()));
}

///< Provide isfinite for RadFadType
bool isfinite(Sacado::Rad::ADvar<Sacado::Fad::DFad<double>> value)
{
    return std::isfinite(static_cast<double>(value.val().val()));
}

template <int dim, typename real>
inline real ManufacturedSolutionSine<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = this->amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= sin( this->frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionAdd<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += this->amplitudes[istate]*sin( this->frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionCosine<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = this->amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= cos( this->frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionExp<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += exp( point[d] );
        assert(isfinite(value));
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionEvenPoly<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    const double poly_max = 7;
    for (int d=0; d<dim; d++) {
        value += pow(point[d] + 0.5, poly_max);
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionPoly<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    for (int d=0; d<dim; d++) {
        const real x = point[d];
        value += 1.0 + x - x*x - x*x*x + x*x*x*x - x*x*x*x*x + x*x*x*x*x*x + 0.001*sin(50*x);
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionAtan<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real val = 1.0;
    for(unsigned int i = 0; i < dim; ++i){
        real x = point[i];
        real val_dim = 0;
        for(unsigned int j = 0; j < n_shocks[i]; ++j){
            // taking the product of function in each direction
            val_dim += atan(S_j[i][j]*(x-x_j[i][j]));
        }
        val *= val_dim;
    }
    return val;
}

template <int dim, typename real>
inline real ManufacturedSolutionBoundaryLayer<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real val = 1.0;
    for(unsigned int d = 0; d < dim; ++d){
        real x = point[d];
        val *= x + (exp(x/epsilon[istate][d])-1.0)/(1.0-exp(1.0/epsilon[istate][d]));
    }
    return val;
}

template <int dim, typename real>
inline real ManufacturedSolutionSShock<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real val = 0.0;
    if(dim==2){
        const real x = point[0], y = point[1];
        // val = 0.75*tanh(2*(sin(5*y)-3*x));
        // val = 0.75*tanh(20*(sin(10*y-5)-6*x+3));
        val = a*tanh(b*sin(c*y + d) + e*x + f);
    }
    return val;
}

template <int dim, typename real>
inline real ManufacturedSolutionQuadratic<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real val = 0.0;
    // f(x,y,z) = a*x^2 + b*y^2 + c*z^2
    for(unsigned int d = 0; d < dim; ++d){
        real x = point[d];
        val += alpha_diag[d]*x*x;
    }
    return val;
}

template <int dim, typename real>
inline real ManufacturedSolutionAlex<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += exp( point[d] ) + sin(point[d] );
        assert(isfinite(value));
    }
    value += this->base_values[istate];
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionNavahBase<dim,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0], y = point[1];
        // // for RANS
        // const real v_tilde = ncm[istate][0] + ncm[istate][1]*cos(ncm[istate][4]*c*x) + ncm[istate][2]*cos(ncm[istate][5]*c*y) + ncm[istate][3]*cos(ncm[istate][6]*c*x)*cos(ncm[istate][6]*c*y);
            
        if(istate==0) {
            // density
            value = ncm[0][0] + ncm[0][1]*sin(ncm[0][4]*c*x) + ncm[0][2]*cos(ncm[0][5]*c*y) + ncm[0][3]*cos(ncm[0][6]*c*x)*cos(ncm[0][6]*c*y);
        }
        if(istate==1) {
            // x-velocity
            value = ncm[1][0] + ncm[1][1]*sin(ncm[1][4]*c*x) + ncm[1][2]*cos(ncm[1][5]*c*y) + ncm[1][3]*cos(ncm[1][6]*c*x)*cos(ncm[1][6]*c*y); 
        }
        if(istate==2) {  
            // y-velocity
            value = ncm[2][0] + ncm[2][1]*cos(ncm[2][4]*c*x) + ncm[2][2]*sin(ncm[2][5]*c*y) + ncm[2][3]*cos(ncm[2][6]*c*x)*cos(ncm[2][6]*c*y);
        }
        if(istate==3) {
            // pressure
            value = ncm[3][0] + ncm[3][1]*cos(ncm[3][4]*c*x) + ncm[3][2]*sin(ncm[3][5]*c*y) + ncm[3][3]*cos(ncm[3][6]*c*x)*cos(ncm[3][6]*c*y);
        }
    }
    return value;
}

template <int dim, typename real>
inline real ManufacturedSolutionNavahBase<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if (dim == 2) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real p   = primitive_value(point,3);

        // convert primitive to conservative solution
        if(istate==0) value = rho; // density
        if(istate==1) value = rho*u; // x-momentum
        if(istate==2) value = rho*v; // y-momentum
        if(istate==3) value = p/(1.4-1.0) + 0.5*rho*(u*u + v*v); // total energy
    }
    return value;
}


template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionSine<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    for (int dim_deri=0; dim_deri<dim; dim_deri++) {
        gradient[dim_deri] = this->amplitudes[istate] * this->frequencies[istate][dim_deri];
        for (int dim_trig=0; dim_trig<dim; dim_trig++) {
            const real angle = this->frequencies[istate][dim_trig] * point[dim_trig];
            if (dim_deri == dim_trig) gradient[dim_deri] *= cos( angle );
            if (dim_deri != dim_trig) gradient[dim_deri] *= sin( angle );
        }
        assert(isfinite(gradient[dim_deri]));
    }
    // Hard-coded is much more readable than the dimensionally generic one
    const real A = this->amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = this->frequencies[istate];
    if (dim==1) {
        const real fx = f[0]*point[0];
        gradient[0] = A*f[0]*cos(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        gradient[0] = A*f[0]*cos(fx)*sin(fy);
        gradient[1] = A*f[1]*sin(fx)*cos(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        gradient[0] = A*f[0]*cos(fx)*sin(fy)*sin(fz);
        gradient[1] = A*f[1]*sin(fx)*cos(fy)*sin(fz);
        gradient[2] = A*f[2]*sin(fx)*sin(fy)*cos(fz);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionAdd<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    const real A = this->amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = this->frequencies[istate];
    if (dim==1) {
        const real fx = f[0]*point[0];
        gradient[0] = A*f[0]*cos(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        gradient[0] = A*f[0]*cos(fx);
        gradient[1] = A*f[1]*cos(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        gradient[0] = A*f[0]*cos(fx);
        gradient[1] = A*f[1]*cos(fy);
        gradient[2] = A*f[2]*cos(fz);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionCosine<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    const real A = this->amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = this->frequencies[istate];
    if (dim==1) {
        const real fx = f[0]*point[0];
        gradient[0] = -A*f[0]*sin(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        gradient[0] = -A*f[0]*sin(fx)*cos(fy);
        gradient[1] = -A*f[1]*cos(fx)*sin(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        gradient[0] = -A*f[0]*sin(fx)*cos(fy)*cos(fz);
        gradient[1] = -A*f[1]*cos(fx)*sin(fy)*cos(fz);
        gradient[2] = -A*f[2]*cos(fx)*cos(fy)*sin(fz);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionExp<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    if (dim==1) {
        gradient[0] = exp(point[0]);
    }
    if (dim==2) {
        gradient[0] = exp(point[0]);
        gradient[1] = exp(point[1]);
    }
    if (dim==3) {
        gradient[0] = exp(point[0]);
        gradient[1] = exp(point[1]);
        gradient[2] = exp(point[2]);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionEvenPoly<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int  /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    const double poly_max = 7;
    if (dim==1) {
        gradient[0] = poly_max*pow(point[0] + 0.5, poly_max-1);
    }
    if (dim==2) {
        gradient[0] = poly_max*pow(point[0] + 0.5, poly_max-1);
        gradient[1] = poly_max*pow(point[1] + 0.5, poly_max-1);
    }
    if (dim==3) {
        gradient[0] = poly_max*pow(point[0] + 0.5, poly_max-1);
        gradient[1] = poly_max*pow(point[1] + 0.5, poly_max-1);
        gradient[2] = poly_max*pow(point[2] + 0.5, poly_max-1);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionPoly<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int  /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    if (dim==1) {
        const real x = point[0];
        gradient[0] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x + 0.050*cos(50*x);
    }
    if (dim==2) {
        real x = point[0];
        gradient[0] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x + 0.050*cos(50*x);
        x = point[1];
        gradient[1] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x + 0.050*cos(50*x);
    }
    if (dim==3) {
        real x = point[0];
        gradient[0] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x;
        x = point[1];
        gradient[1] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x;
        x = point[2];
        gradient[2] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x;
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionAtan<dim,real>
::gradient(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    for(unsigned int k = 0; k < dim; ++k){
        // taking the k^th derivative
        real grad_dim = 1;
        for(unsigned int i = 0; i < dim; ++i){
            real x = point[i];
            real val_dim = 0;
            for(unsigned int j = 0; j < n_shocks[i]; ++j){
                if(i==k){
                    // taking the derivative dimension
                    real coeff = S_j[i][j]*(x-x_j[i][j]);
                    val_dim += S_j[i][j]/(pow(coeff,2)+1);
                }else{
                    // value product unaffected
                    val_dim += atan(S_j[i][j]*(x-x_j[i][j]));
                }
            }
            grad_dim *= val_dim;
        }
        gradient[k] = grad_dim;
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionBoundaryLayer<dim,real>
::gradient(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    if(dim == 1){
        const real x = point[0];
        gradient[0] = (1 + (exp(x/epsilon[istate][0])/epsilon[istate][0])/(1.0-exp(1.0/epsilon[istate][0])));
    }else if(dim == 2){
        const real x = point[0], y = point[1];
        gradient[0] = (1 + (exp(x/epsilon[istate][0])/epsilon[istate][0])/(1.0-exp(1.0/epsilon[istate][0])))
                    * (y + (exp(y/epsilon[istate][1])-1.0)               /(1.0-exp(1.0/epsilon[istate][1])));
        gradient[1] = (x + (exp(x/epsilon[istate][0])-1.0)               /(1.0-exp(1.0/epsilon[istate][0])))
                    * (1 + (exp(y/epsilon[istate][1])/epsilon[istate][1])/(1.0-exp(1.0/epsilon[istate][1])));
    }else if(dim == 3){
        const real x = point[0], y = point[1], z = point[2];
        gradient[0] = (1 + (exp(x/epsilon[istate][0])/epsilon[istate][0])/(1.0-exp(1.0/epsilon[istate][0])))
                    * (y + (exp(y/epsilon[istate][1])-1.0)               /(1.0-exp(1.0/epsilon[istate][1])))
                    * (z + (exp(z/epsilon[istate][2])-1.0)               /(1.0-exp(1.0/epsilon[istate][2])));
        gradient[1] = (x + (exp(x/epsilon[istate][0])-1.0)               /(1.0-exp(1.0/epsilon[istate][0])))
                    * (1 + (exp(y/epsilon[istate][1])/epsilon[istate][1])/(1.0-exp(1.0/epsilon[istate][1])))
                    * (z + (exp(z/epsilon[istate][2])-1.0)               /(1.0-exp(1.0/epsilon[istate][2])));
        gradient[2] = (x + (exp(x/epsilon[istate][0])-1.0)               /(1.0-exp(1.0/epsilon[istate][0])))
                    * (y + (exp(y/epsilon[istate][1])-1.0)               /(1.0-exp(1.0/epsilon[istate][1])))
                    * (1 + (exp(z/epsilon[istate][2])/epsilon[istate][2])/(1.0-exp(1.0/epsilon[istate][2])));
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionSShock<dim,real>
::gradient(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    if(dim == 2){
        const real x = point[0], y = point[1];
        // gradient[0] = -4.5*pow(cosh(6*x-2*sin(5*y)),-2);  
        // gradient[1] =  7.5*pow(cosh(6*x-2*sin(5*y)),-2)*cos(5*y);
        // gradient[0] = -90*pow(cosh(-120*x-20*sin(5-10*y)+60),-2);
        // gradient[1] = 150*pow(cosh(-120*x-20*sin(5-10*y)+60),-2)*cos(5-10*y);

        const real denominator = pow(cosh(f + e*x + b*sin(d + c*y)), -2);
        gradient[0] =              a*e*denominator;
        gradient[1] = a*b*c*cos(d+c*y)*denominator; 
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionQuadratic<dim,real>
::gradient(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    for(unsigned int d = 0; d < dim; ++d){
        // dF = <2ax, 2by, 2cz>
        const real x = point[d];
        gradient[d] = 2*alpha_diag[d]*x;
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionAlex<dim,real>
::gradient(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;
    for(unsigned int d = 0; d < dim; ++d){
        gradient[d] = exp(point[d]) + cos(point[d]);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionNavahBase<dim,real>
::primitive_gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    // Gradients of primitive variables 
    if (dim == 2) {
        const real x = point[0], y = point[1];

        if(istate==0) {
            // density
            gradient[0] =  ncm[0][4]*c*ncm[0][1]*cos(ncm[0][4]*c*x) - ncm[0][6]*c*ncm[0][3]*sin(ncm[0][6]*c*x)*cos(ncm[0][6]*c*y); // dx
            gradient[1] = -ncm[0][5]*c*ncm[0][2]*sin(ncm[0][5]*c*y) - ncm[0][6]*c*ncm[0][3]*cos(ncm[0][6]*c*x)*sin(ncm[0][6]*c*y); // dy
        }
        if(istate==1) {
            // x-velocity
            gradient[0] =  ncm[1][4]*c*ncm[1][1]*cos(ncm[1][4]*c*x) - ncm[1][6]*c*ncm[1][3]*sin(ncm[1][6]*c*x)*cos(ncm[1][6]*c*y); // dx
            gradient[1] = -ncm[1][5]*c*ncm[1][2]*sin(ncm[1][5]*c*y) - ncm[1][6]*c*ncm[1][3]*cos(ncm[1][6]*c*x)*sin(ncm[1][6]*c*y); // dy
        }
        if(istate==2) {
            // y-velocity
            gradient[0] = -ncm[2][4]*c*ncm[2][1]*sin(ncm[2][4]*c*x) - ncm[2][6]*c*ncm[2][3]*sin(ncm[2][6]*c*x)*cos(ncm[2][6]*c*y); // dx
            gradient[1] =  ncm[2][5]*c*ncm[2][2]*cos(ncm[2][5]*c*y) - ncm[2][6]*c*ncm[2][3]*cos(ncm[2][6]*c*x)*sin(ncm[2][6]*c*y); // dy
        }
        if(istate==3) {
            // pressure
            gradient[0] = -ncm[3][4]*c*ncm[3][1]*sin(ncm[3][4]*c*x) - ncm[3][6]*c*ncm[3][3]*sin(ncm[3][6]*c*x)*cos(ncm[3][6]*c*y); // dx
            gradient[1] =  ncm[3][5]*c*ncm[3][2]*cos(ncm[3][5]*c*y) - ncm[3][6]*c*ncm[3][3]*cos(ncm[3][6]*c*x)*sin(ncm[3][6]*c*y); // dy
        }
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionNavahBase<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;

    if (dim == 2) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        // const real p   = primitive_value(point,3);
        const dealii::Tensor<1,dim,real> rho_grad = primitive_gradient(point,0);
        const dealii::Tensor<1,dim,real> u_grad   = primitive_gradient(point,1);
        const dealii::Tensor<1,dim,real> v_grad   = primitive_gradient(point,2);
        const dealii::Tensor<1,dim,real> p_grad   = primitive_gradient(point,3);
        
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
            // total energy
            for(int d=0; d<dim; d++) {
                gradient[d] = p_grad[d]/(1.4-1.0) + 0.5*rho_grad[d]*(u*u + v*v) + rho*(u*u_grad[d]+v*v_grad[d]);
            }
        }
    }
    return gradient;
}


template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionSine<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    // Hard-coded is much more readable than the dimensionally generic one
    const real A = this->amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = this->frequencies[istate];
    if (dim==1) {
        const real fx = f[0]*point[0];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx)*sin(fy);
        hessian[0][1] =  A*f[0]*f[1]*cos(fx)*cos(fy);

        hessian[1][0] =  A*f[1]*f[0]*cos(fx)*cos(fy);
        hessian[1][1] = -A*f[1]*f[1]*sin(fx)*sin(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx)*sin(fy)*sin(fz);
        hessian[0][1] =  A*f[0]*f[1]*cos(fx)*cos(fy)*sin(fz);
        hessian[0][2] =  A*f[0]*f[2]*cos(fx)*sin(fy)*cos(fz);
        
        hessian[1][0] =  A*f[1]*f[0]*cos(fx)*cos(fy)*sin(fz);
        hessian[1][1] = -A*f[1]*f[1]*sin(fx)*sin(fy)*sin(fz);
        hessian[1][2] =  A*f[1]*f[2]*sin(fx)*cos(fy)*cos(fz);
        
        hessian[2][0] =  A*f[2]*f[0]*cos(fx)*sin(fy)*cos(fz);
        hessian[2][1] =  A*f[2]*f[1]*sin(fx)*cos(fy)*cos(fz);
        hessian[2][2] = -A*f[2]*f[2]*sin(fx)*sin(fy)*sin(fz);
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionAdd<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    const real A = this->amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = this->frequencies[istate];
    if (dim==1) {
        const real fx = f[0]*point[0];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
        hessian[0][1] =  0.0;

        hessian[1][0] =  0.0;
        hessian[1][1] = -A*f[1]*f[1]*sin(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
        hessian[0][1] =  0.0;
        hessian[0][2] =  0.0;
        
        hessian[1][0] =  0.0;
        hessian[1][1] = -A*f[1]*f[1]*sin(fy);
        hessian[1][2] =  0.0;
        
        hessian[2][0] =  0.0;
        hessian[2][1] =  0.0;
        hessian[2][2] = -A*f[2]*f[2]*sin(fz);
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionCosine<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    const real A = this->amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = this->frequencies[istate];
    if (dim==1) {
        const real fx = f[0]*point[0];
        hessian[0][0] = -A*f[0]*f[0]*cos(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        hessian[0][0] = -A*f[0]*f[0]*cos(fx)*cos(fy);
        hessian[0][1] =  A*f[0]*f[1]*sin(fx)*sin(fy);

        hessian[1][0] =  A*f[1]*f[0]*sin(fx)*sin(fy);
        hessian[1][1] = -A*f[1]*f[1]*cos(fx)*cos(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        hessian[0][0] = -A*f[0]*f[0]*cos(fx)*cos(fy)*cos(fz);
        hessian[0][1] =  A*f[0]*f[1]*sin(fx)*sin(fy)*cos(fz);
        hessian[0][2] =  A*f[0]*f[2]*sin(fx)*cos(fy)*sin(fz);
        
        hessian[1][0] =  A*f[1]*f[0]*sin(fx)*sin(fy)*cos(fz);
        hessian[1][1] = -A*f[1]*f[1]*cos(fx)*cos(fy)*cos(fz);
        hessian[1][2] =  A*f[1]*f[2]*cos(fx)*sin(fy)*sin(fz);
        
        hessian[2][0] =  A*f[2]*f[0]*sin(fx)*cos(fy)*sin(fz);
        hessian[2][1] =  A*f[2]*f[1]*cos(fx)*sin(fy)*sin(fz);
        hessian[2][2] = -A*f[2]*f[2]*cos(fx)*cos(fy)*cos(fz);
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionExp<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int  /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    if (dim==1) {
        hessian[0][0] = exp(point[0]);
    }
    if (dim==2) {
        hessian[0][0] = exp(point[0]);
        hessian[0][1] = 0.0;

        hessian[1][0] = 0.0;
        hessian[1][1] = exp(point[1]);
    }
    if (dim==3) {
        hessian[0][0] = exp(point[0]);
        hessian[0][1] = 0.0;
        hessian[0][2] = 0.0;
        
        hessian[1][0] = 0.0;
        hessian[1][1] = exp(point[1]);
        hessian[1][2] = 0.0;
        
        hessian[2][0] = 0.0;
        hessian[2][1] = 0.0;
        hessian[2][2] = exp(point[2]);
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionEvenPoly<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int  /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    const double poly_max = 7;
    if (dim==1) {
        hessian[0][0] = poly_max*poly_max*pow(point[0] + 0.5, poly_max-2);
    }
    if (dim==2) {
        hessian[0][0] = poly_max*poly_max*pow(point[0] + 0.5, poly_max-2);
        hessian[0][1] = 0.0;

        hessian[1][0] = 0.0;
        hessian[1][1] = poly_max*poly_max*pow(point[1] + 0.5, poly_max-2);
    }
    if (dim==3) {
        hessian[0][0] = poly_max*poly_max*pow(point[0] + 0.5, poly_max-2);
        hessian[0][1] = 0.0;
        hessian[0][2] = 0.0;
        
        hessian[1][0] = 0.0;
        hessian[1][1] = poly_max*poly_max*pow(point[1] + 0.5, poly_max-2);
        hessian[1][2] = 0.0;
        
        hessian[2][0] = 0.0;
        hessian[2][1] = 0.0;
        hessian[2][2] = poly_max*poly_max*pow(point[2] + 0.5, poly_max-2);
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionPoly<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int  /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    if (dim==1) {
        const real x = point[0];
        hessian[0][0] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
    }
    if (dim==2) {
        real x = point[0];
        hessian[0][0] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
        x = point[1];
        hessian[1][1] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
    }
    if (dim==3) {
        real x = point[0];
        hessian[0][0] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
        x = point[1];
        hessian[1][1] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
        x = point[2];
        hessian[2][2] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionAtan<dim,real>
::hessian(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hes;

    for(unsigned int k1 = 0; k1 < dim; ++k1){
        // taking the k1^th derivative
        for(unsigned int k2 = 0; k2 < dim; ++k2){
            // taking the k2^th derivative
            real hes_dim = 1;
            for(unsigned int i = 0; i < dim; ++i){
                real x = point[i];
                real val_dim = 0;
                for(unsigned int j = 0; j < n_shocks[i]; ++j){
                    if(i == k1 && i == k2){
                        // taking the second derivative in this dim
                        real coeff = S_j[i][j]*(x-x_j[i][j]);
                        val_dim += -2.0*pow(S_j[i][j],2)*coeff/pow(pow(coeff,2)+1,2);
                    }else if(i == k1 || i == k2){
                        // taking the first derivative in this dim
                        real coeff = S_j[i][j]*(x-x_j[i][j]);
                        val_dim += S_j[i][j]/(pow(coeff,2)+1);
                    }else{
                        // taking the value in this dim
                        val_dim += atan(S_j[i][j]*(x-x_j[i][j]));
                    }
                }
                hes_dim *= val_dim;
            }
            hes[k1][k2] = hes_dim;
        }
    }

    return hes;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionBoundaryLayer<dim,real>
::hessian(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    if (dim==1) {
        const real x = point[0];
        hessian[0][0] = (exp(x/epsilon[istate][0])/pow(epsilon[istate][0],2)/(1.0-exp(1.0/epsilon[istate][0])));
    }
    if (dim==2) {
        real x = point[0], y = point[1];
        hessian[0][0] = (exp(x/epsilon[istate][0])/pow(epsilon[istate][0],2)/(1.0-exp(1.0/epsilon[istate][0])))
                      * (y + (exp(y/epsilon[istate][1])-1.0)                /(1.0-exp(1.0/epsilon[istate][1])));
        hessian[0][1] = (1 + (exp(x/epsilon[istate][0])/epsilon[istate][0]) /(1.0-exp(1.0/epsilon[istate][0])))
                      * (1 + (exp(y/epsilon[istate][1])/epsilon[istate][1]) /(1.0-exp(1.0/epsilon[istate][1])));
        
        hessian[1][0] = hessian[0][1];
        hessian[1][1] = (x + (exp(x/epsilon[istate][0])-1.0)                /(1.0-exp(1.0/epsilon[istate][0])))
                      * (exp(y/epsilon[istate][1])/pow(epsilon[istate][1],2)/(1.0-exp(1.0/epsilon[istate][1])));
    }
    if (dim==3) {
        real x = point[0], y = point[1], z = point[2];
        hessian[0][0] = (exp(x/epsilon[istate][0])/pow(epsilon[istate][0],2)/(1.0-exp(1.0/epsilon[istate][0])))
                      * (y + (exp(y/epsilon[istate][1])-1.0)                /(1.0-exp(1.0/epsilon[istate][1])))
                      * (z + (exp(z/epsilon[istate][2])-1.0)                /(1.0-exp(1.0/epsilon[istate][2])));
        hessian[0][1] = (1 + (exp(x/epsilon[istate][0])/epsilon[istate][0]) /(1.0-exp(1.0/epsilon[istate][0])))
                      * (1 + (exp(y/epsilon[istate][1])/epsilon[istate][1]) /(1.0-exp(1.0/epsilon[istate][1])))
                      * (z + (exp(z/epsilon[istate][2])-1.0)                /(1.0-exp(1.0/epsilon[istate][2])));
        hessian[0][2] = (1 + (exp(x/epsilon[istate][0])/epsilon[istate][0]) /(1.0-exp(1.0/epsilon[istate][0])))
                      * (y + (exp(y/epsilon[istate][1])-1.0)                /(1.0-exp(1.0/epsilon[istate][1])))
                      * (1 + (exp(z/epsilon[istate][2])/epsilon[istate][2]) /(1.0-exp(1.0/epsilon[istate][2])));

        hessian[1][0] = hessian[0][1];
        hessian[1][1] = (x + (exp(x/epsilon[istate][0])-1.0)                /(1.0-exp(1.0/epsilon[istate][0])))
                      * (exp(y/epsilon[istate][1])/pow(epsilon[istate][1],2)/(1.0-exp(1.0/epsilon[istate][1])))
                      * (z + (exp(z/epsilon[istate][2])-1.0)                /(1.0-exp(1.0/epsilon[istate][2])));
        hessian[1][2] = (x + (exp(x/epsilon[istate][0])-1.0)                /(1.0-exp(1.0/epsilon[istate][0])))
                      * (1 + (exp(y/epsilon[istate][1])/epsilon[istate][1]) /(1.0-exp(1.0/epsilon[istate][1])))
                      * (1 + (exp(z/epsilon[istate][2])/epsilon[istate][2]) /(1.0-exp(1.0/epsilon[istate][2])));

        hessian[2][0] = hessian[0][2];
        hessian[2][1] = hessian[2][1];
        hessian[2][2] = (x + (exp(x/epsilon[istate][0])-1.0)                /(1.0-exp(1.0/epsilon[istate][0])))
                      * (y + (exp(y/epsilon[istate][1])-1.0)                /(1.0-exp(1.0/epsilon[istate][1])))
                      * (exp(z/epsilon[istate][2])/pow(epsilon[istate][2],2)/(1.0-exp(1.0/epsilon[istate][2])));
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionSShock<dim,real>
::hessian(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    if (dim==2) {
        const real x = point[0], y = point[1];
        // hessian[0][0] =  54*tanh(6*x-2*sin(5*y))*pow(cosh(6*x-2*sin(5*y)),-2);
        // hessian[0][1] = -90*tanh(6*x-2*sin(5*y))*pow(cosh(6*x-2*sin(5*y)),-2)*cos(5*y);

        // hessian[1][0] = hessian[1][0];
        // hessian[1][1] = pow(cosh(6*x-2*sin(5*y)),-2)*(-37.5*sin(5*y)+150*pow(cos(5*y),2)*tanh(6*x-2*sin(5*y)));

        // hessian[0][0] =  21600*pow(cosh(20*(-3+6*x+sin(5-10*y))),2)*tanh(20*(-3+6*x+sin(5-10*y)));
        // hessian[0][1] = -36000*pow(cosh(20*(-3+6*x+sin(5-10*y))),2)*tanh(20*(-3+6*x+sin(5-10*y)))*cos(5-10*y);

        // hessian[1][0] = hessian[0][1];
        // hessian[1][1] =   1500*pow(cosh(20*(-3+6*x+sin(5-10*y))),2)*(40*pow(cos(5-10*y),2)*tanh(20*(-3+6*x+sin(5-10*y)))+sin(5-10*y));
    
        const real component   = f + e*x + b*sin(d+c*y);
        const real numerator   = sinh(component); 
        const real denominator = pow(cosh(component), -3);

        hessian[0][0] =              -2*a*e*e*numerator*denominator;
        hessian[0][1] = -2*a*b*c*e*cos(d+c*y)*numerator*denominator;

        hessian[1][0] = hessian[0][1];
        hessian[1][1] = -a*b*c*c*pow(cosh(component), -2)*(2*b*pow(cos(c*y + d),2)*tanh(component) + sin(c*y + d));
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionQuadratic<dim,real>
::hessian(const dealii::Point<dim,real> &/* point */, const unsigned int /* istate */) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    for(unsigned int i = 0; i < dim; ++i){
        for(unsigned int j = 0; j < dim; ++j){
            if(i == j){
                hessian[i][i] = 2*alpha_diag[i];
            }else{
                hessian[i][j] = 0.0;
            }
        }
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionAlex<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int  /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    for(int idim=0; idim<dim; idim++){
        for(int jdim=0; jdim<dim; jdim++){
            if(idim == jdim)
                hessian[idim][jdim] = exp(point[idim]) - sin(point[idim]); 
            else
                hessian[idim][jdim] = 0.0;
        }
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionNavahBase<dim,real>
::primitive_hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    if (dim == 2) {
        const real x = point[0], y = point[1];

        if(istate==0) {
            // density
            hessian[0][0] = -ncm[0][4]*c*ncm[0][4]*c*ncm[0][1]*sin(ncm[0][4]*c*x) - ncm[0][6]*c*ncm[0][6]*c*ncm[0][3]*cos(ncm[0][6]*c*x)*cos(ncm[0][6]*c*y); // dxdx
            hessian[0][1] =  ncm[0][6]*c*ncm[0][6]*c*ncm[0][3]*sin(ncm[0][6]*c*x)*sin(ncm[0][6]*c*y); // dxdy
            hessian[1][0] =  hessian[0][1]; // dydx
            hessian[1][1] = -ncm[0][5]*c*ncm[0][5]*c*ncm[0][2]*cos(ncm[0][5]*c*y) - ncm[0][6]*c*ncm[0][6]*c*ncm[0][3]*cos(ncm[0][6]*c*x)*cos(ncm[0][6]*c*y); // dydy
        }
        if(istate==1) {
            // x-velocity
            hessian[0][0] = -ncm[1][4]*c*ncm[1][4]*c*ncm[1][1]*sin(ncm[1][4]*c*x) - ncm[1][6]*c*ncm[1][6]*c*ncm[1][3]*cos(ncm[1][6]*c*x)*cos(ncm[1][6]*c*y); // dxdx
            hessian[0][1] =  ncm[1][6]*c*ncm[1][6]*c*ncm[1][3]*sin(ncm[1][6]*c*x)*sin(ncm[1][6]*c*y); // dxdy
            hessian[1][0] =  hessian[0][1]; // dydx
            hessian[1][1] = -ncm[1][5]*c*ncm[1][5]*c*ncm[1][2]*cos(ncm[1][5]*c*y) - ncm[1][6]*c*ncm[1][6]*c*ncm[1][3]*cos(ncm[1][6]*c*x)*cos(ncm[1][6]*c*y); // dydy
        }
        if(istate==2) {
            // y-velocity
            hessian[0][0] = -ncm[2][4]*c*ncm[2][4]*c*ncm[2][1]*cos(ncm[2][4]*c*x) - ncm[2][6]*c*ncm[2][6]*c*ncm[2][3]*cos(ncm[2][6]*c*x)*cos(ncm[2][6]*c*y); // dxdx
            hessian[0][1] =  ncm[2][6]*c*ncm[2][6]*c*ncm[2][3]*sin(ncm[2][6]*c*x)*sin(ncm[2][6]*c*y); // dxdy
            hessian[1][0] =  hessian[0][1]; // dydx
            hessian[1][1] = -ncm[2][5]*c*ncm[2][5]*c*ncm[2][2]*sin(ncm[2][5]*c*y) - ncm[2][6]*c*ncm[2][6]*c*ncm[2][3]*cos(ncm[2][6]*c*x)*cos(ncm[2][6]*c*y); // dydy
        }
        if(istate==3) {
            // pressure
            hessian[0][0] = -ncm[3][4]*c*ncm[3][4]*c*ncm[3][1]*cos(ncm[3][4]*c*x) - ncm[3][6]*c*ncm[3][6]*c*ncm[3][3]*cos(ncm[3][6]*c*x)*cos(ncm[3][6]*c*y); // dxdx
            hessian[0][1] =  ncm[3][6]*c*ncm[3][6]*c*ncm[3][3]*sin(ncm[3][6]*c*x)*sin(ncm[3][6]*c*y); // dxdy
            hessian[1][0] =  hessian[0][1]; // dydx
            hessian[1][1] = -ncm[3][5]*c*ncm[3][5]*c*ncm[3][2]*sin(ncm[3][5]*c*y) - ncm[3][6]*c*ncm[3][6]*c*ncm[3][3]*cos(ncm[3][6]*c*x)*cos(ncm[3][6]*c*y); // dydy
        }
    }
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionNavahBase<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    if (dim == 2) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        // const real p   = primitive_value(point,3);
        const dealii::Tensor<1,dim,real> rho_grad = primitive_gradient(point,0);
        const dealii::Tensor<1,dim,real> u_grad   = primitive_gradient(point,1);
        const dealii::Tensor<1,dim,real> v_grad   = primitive_gradient(point,2);
        // const dealii::Tensor<1,dim,real> p_grad   = primitive_gradient(point,3);
        const dealii::SymmetricTensor<2,dim,real> rho_hess = primitive_hessian(point,0);
        const dealii::SymmetricTensor<2,dim,real> u_hess   = primitive_hessian(point,1);
        const dealii::SymmetricTensor<2,dim,real> v_hess   = primitive_hessian(point,2);
        const dealii::SymmetricTensor<2,dim,real> p_hess   = primitive_hessian(point,3);

        // convert to primitive to hessian of conservative variables using product rule
        if(istate==0) {
            // density
            for(int i=0; i<dim; i++) { 
                for(int j=0; j<dim; j++) { 
                    hessian[i][j] = rho_hess[i][j];
                }
            }
        }
        if(istate==1) {
            // x-momentum
            for(int i=0; i<dim; i++) { 
                for(int j=0; j<dim; j++) { 
                    hessian[i][j] = u_grad[j]*rho_grad[i] + u*rho_hess[i][j] + rho_grad[j]*u_grad[i] + rho*u_hess[i][j];
                }
            }
        }
        if(istate==2) {
            // y-momentum
            for(int i=0; i<dim; i++) { 
                for(int j=0; j<dim; j++) { 
                    hessian[i][j] = v_grad[j]*rho_grad[i] + v*rho_hess[i][j] + rho_grad[j]*v_grad[i] + rho*v_hess[i][j];
                }
            }
        }
        if(istate==3) {
            // total energy
            for(int i=0; i<dim; i++) { 
                for(int j=0; j<dim; j++) { 
                    hessian[i][j]  = p_hess[i][j]/(1.4-1.0) + (u*u_grad[j]+v*v_grad[j])*rho_grad[i] + 0.5*(u*u + v*v)*rho_hess[i][j];
                    hessian[i][j] += rho_grad[j]*(u*u_grad[i]+v*v_grad[i]);
                    hessian[i][j] += rho*(u_grad[j]*u_grad[i] + u*u_hess[i][j] + v_grad[j]*v_grad[i] + v*v_hess[i][j]);
                }
            }
        }
    }
    return hessian;
}

template <int dim, typename real>
ManufacturedSolutionFunction<dim,real>
::ManufacturedSolutionFunction (const unsigned int nstate)
    :
    dealii::Function<dim,real>(nstate)
    , nstate(nstate)
    , base_values(nstate)
    , amplitudes(nstate)
    , frequencies(nstate)
{
    const double pi = atan(1)*4.0;
    //const double ee = exp(1);

    for (int s=0; s<(int)nstate; s++) {
        base_values[s] = 1+(s+1.0)/nstate;
        base_values[nstate-1] = 10;
        amplitudes[s] = 0.2*base_values[s]*sin((static_cast<double>(nstate)-s)/nstate);
        for (int d=0; d<dim; d++) {
            //frequencies[s][d] = 2.0 + sin(0.1+s*0.5+d*0.2) *  pi / 2.0;
            frequencies[s][d] = 2.0 + sin(0.1+s*0.5+d*0.2) *  pi / 2.0;
        }
    
    }
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,real>
::gradient_fd (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    const double eps=1e-6;
    for (int dim_deri=0; dim_deri<dim; dim_deri++) {
        dealii::Point<dim,real> pert_p = point;
        dealii::Point<dim,real> pert_m = point;
        pert_p[dim_deri] += eps;
        pert_m[dim_deri] -= eps;
        const real value_p = value(pert_p,istate);
        const real value_m = value(pert_m,istate);
        gradient[dim_deri] = (value_p - value_m) / (2*eps);
    }
    return gradient;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionFunction<dim,real>
::hessian_fd (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    const double eps=1e-4;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=d1; d2<dim; d2++) {
            dealii::Point<dim,real> pert_p_p = point;
            dealii::Point<dim,real> pert_p_m = point;
            dealii::Point<dim,real> pert_m_p = point;
            dealii::Point<dim,real> pert_m_m = point;

            pert_p_p[d1] += (+eps); pert_p_p[d2] += (+eps);
            pert_p_m[d1] += (+eps); pert_p_m[d2] += (-eps);
            pert_m_p[d1] += (-eps); pert_m_p[d2] += (+eps);
            pert_m_m[d1] += (-eps); pert_m_m[d2] += (-eps);

            const real valpp = value(pert_p_p, istate);
            const real valpm = value(pert_p_m, istate);
            const real valmp = value(pert_m_p, istate);
            const real valmm = value(pert_m_m, istate);

            hessian[d1][d2] = (valpp - valpm - valmp + valmm) / (4*eps*eps);
        }
    }
    return hessian;
}

template <int dim, typename real>
void ManufacturedSolutionFunction<dim,real>
::vector_gradient (
    const dealii::Point<dim,real> &p,
    std::vector<dealii::Tensor<1,dim, real> > &gradients) const
{
    for (unsigned int i = 0; i < nstate; ++i)
        gradients[i] = gradient(p, i);
}


template <int dim, typename real>
inline std::vector<real> ManufacturedSolutionFunction<dim,real>
::stdvector_values (const dealii::Point<dim,real> &point) const
{
    std::vector<real> values(nstate);
    for (unsigned int s=0; s<nstate; s++) { values[s] = value(point, s); }
    return values;
}

template <int dim, typename real>
std::shared_ptr< ManufacturedSolutionFunction<dim,real> > 
ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(
    Parameters::AllParameters const *const param, 
    int                                    nstate)
{
    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
    ManufacturedSolutionEnum solution_type = param->manufactured_convergence_study_param.manufactured_solution_param.manufactured_solution_type;

    return create_ManufacturedSolution(solution_type, nstate);
}

template <int dim, typename real>
std::shared_ptr< ManufacturedSolutionFunction<dim,real> >
ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(
    Parameters::ManufacturedSolutionParam::ManufacturedSolutionType solution_type,
    int                                                                     nstate)
{
    if(solution_type == ManufacturedSolutionEnum::sine_solution){
        return std::make_shared<ManufacturedSolutionSine<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::cosine_solution){
        return std::make_shared<ManufacturedSolutionCosine<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::additive_solution){
        return std::make_shared<ManufacturedSolutionAdd<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::exp_solution){
        return std::make_shared<ManufacturedSolutionExp<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::poly_solution){
        return std::make_shared<ManufacturedSolutionPoly<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::even_poly_solution){
        return std::make_shared<ManufacturedSolutionEvenPoly<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::atan_solution){
        return std::make_shared<ManufacturedSolutionAtan<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::boundary_layer_solution){
        return std::make_shared<ManufacturedSolutionBoundaryLayer<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::s_shock_solution && dim == 2){
        return std::make_shared<ManufacturedSolutionSShock<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::quadratic_solution){
        return std::make_shared<ManufacturedSolutionQuadratic<dim,real>>(nstate);
    }else if(solution_type == ManufacturedSolutionEnum::Alex_solution){
        return std::make_shared<ManufacturedSolutionAlex<dim,real>>(nstate);

    }else if(solution_type == ManufacturedSolutionEnum::navah_solution_1){
        if constexpr((dim==2) /*&& (nstate==dim+2)*/) {
            return std::make_shared<ManufacturedSolutionNavah_MS1<dim,real>>(nstate);
        }
    }else if(solution_type == ManufacturedSolutionEnum::navah_solution_2){
        if constexpr((dim==2) /*&& (nstate==dim+2)*/) {
            return std::make_shared<ManufacturedSolutionNavah_MS2<dim,real>>(nstate);
        }
    }else if(solution_type == ManufacturedSolutionEnum::navah_solution_3){
        if constexpr((dim==2) /*&& (nstate==dim+2)*/) {
            return std::make_shared<ManufacturedSolutionNavah_MS3<dim,real>>(nstate);
        }
    }else if(solution_type == ManufacturedSolutionEnum::navah_solution_4){
        if constexpr((dim==2) /*&& (nstate==dim+2)*/) {
            return std::make_shared<ManufacturedSolutionNavah_MS4<dim,real>>(nstate);
        }
    }else if(solution_type == ManufacturedSolutionEnum::navah_solution_5){
        if constexpr((dim==2) /*&& (nstate==dim+2)*/) {
            return std::make_shared<ManufacturedSolutionNavah_MS5<dim,real>>(nstate);
        }
    }else{
        std::cout << "Invalid Manufactured Solution." << std::endl;
    }

    return nullptr;
}

using FadType = Sacado::Fad::DFad<double>; ///< Sacado AD type for first derivatives.
using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

static constexpr int dimForwardAD = 1; ///< Size of the forward vector mode for CoDiPack.
static constexpr int dimReverseAD = 1; ///< Size of the reverse vector mode for CoDiPack.

using codi_FadType = codi::RealForwardGen<double, codi::Direction<double,dimForwardAD>>; ///< Tapeless forward mode.
//using codi_FadType = codi::RealForwardGen<double, codi::DirectionVar<double>>;

using codi_JacobianComputationType = codi::RealReverseIndexVec<dimReverseAD>; ///< Reverse mode type for Jacobian computation using TapeHelper.
using codi_HessianComputationType = codi::RealReversePrimalIndexGen< codi::RealForwardVec<dimForwardAD>,
                                                  codi::Direction< codi::RealForwardVec<dimForwardAD>, dimReverseAD>
                                                >; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.
//using RadFadType = Sacado::Rad::ADvar<FadType>; ///< Sacado AD type that allows 2nd derivatives.
//using RadFadType = codi_JacobianComputationType; ///< Reverse only mode that only allows Jacobian computation.
using RadType = codi_JacobianComputationType; ///< CoDiPaco reverse-AD type for first derivatives.
using RadFadType = codi_HessianComputationType; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.

template class ManufacturedSolutionFunction<PHILIP_DIM,double>;
template class ManufacturedSolutionFunction<PHILIP_DIM,FadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM,RadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM,RadFadType>;

template class ManufacturedSolutionSine<PHILIP_DIM,double>;
template class ManufacturedSolutionSine<PHILIP_DIM,FadType>;
template class ManufacturedSolutionSine<PHILIP_DIM,RadType>;
template class ManufacturedSolutionSine<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionSine<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionCosine<PHILIP_DIM,double>;
template class ManufacturedSolutionCosine<PHILIP_DIM,FadType>;
template class ManufacturedSolutionCosine<PHILIP_DIM,RadType>;
template class ManufacturedSolutionCosine<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionCosine<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionAdd<PHILIP_DIM,double>;
template class ManufacturedSolutionAdd<PHILIP_DIM,FadType>;
template class ManufacturedSolutionAdd<PHILIP_DIM,RadType>;
template class ManufacturedSolutionAdd<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionAdd<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionExp<PHILIP_DIM,double>;
template class ManufacturedSolutionExp<PHILIP_DIM,FadType>;
template class ManufacturedSolutionExp<PHILIP_DIM,RadType>;
template class ManufacturedSolutionExp<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionExp<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionPoly<PHILIP_DIM,double>;
template class ManufacturedSolutionPoly<PHILIP_DIM,FadType>;
template class ManufacturedSolutionPoly<PHILIP_DIM,RadType>;
template class ManufacturedSolutionPoly<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionPoly<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionEvenPoly<PHILIP_DIM,double>;
template class ManufacturedSolutionEvenPoly<PHILIP_DIM,FadType>;
template class ManufacturedSolutionEvenPoly<PHILIP_DIM,RadType>;
template class ManufacturedSolutionEvenPoly<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionEvenPoly<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionAtan<PHILIP_DIM,double>;
template class ManufacturedSolutionAtan<PHILIP_DIM,FadType>;
template class ManufacturedSolutionAtan<PHILIP_DIM,RadType>;
template class ManufacturedSolutionAtan<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionAtan<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionBoundaryLayer<PHILIP_DIM,double>;
template class ManufacturedSolutionBoundaryLayer<PHILIP_DIM,FadType>;
template class ManufacturedSolutionBoundaryLayer<PHILIP_DIM,RadType>;
template class ManufacturedSolutionBoundaryLayer<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionBoundaryLayer<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionSShock<PHILIP_DIM,double>;
template class ManufacturedSolutionSShock<PHILIP_DIM,FadType>;
template class ManufacturedSolutionSShock<PHILIP_DIM,RadType>;
template class ManufacturedSolutionSShock<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionSShock<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionQuadratic<PHILIP_DIM,double>;
template class ManufacturedSolutionQuadratic<PHILIP_DIM,FadType>;
template class ManufacturedSolutionQuadratic<PHILIP_DIM,RadType>;
template class ManufacturedSolutionQuadratic<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionQuadratic<PHILIP_DIM,RadFadType>;

// Ask Doug: Instantiate for "2" directly instead of PHILIP_DIM ?? SShock is only for 2 but instantiated for PHILIP_DIM
template class ManufacturedSolutionNavahBase<PHILIP_DIM,double>;
template class ManufacturedSolutionNavahBase<PHILIP_DIM,FadType>;
template class ManufacturedSolutionNavahBase<PHILIP_DIM,RadType>;
template class ManufacturedSolutionNavahBase<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionNavahBase<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionNavah_MS1<PHILIP_DIM,double>;
template class ManufacturedSolutionNavah_MS1<PHILIP_DIM,FadType>;
template class ManufacturedSolutionNavah_MS1<PHILIP_DIM,RadType>;
template class ManufacturedSolutionNavah_MS1<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionNavah_MS1<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionNavah_MS2<PHILIP_DIM,double>;
template class ManufacturedSolutionNavah_MS2<PHILIP_DIM,FadType>;
template class ManufacturedSolutionNavah_MS2<PHILIP_DIM,RadType>;
template class ManufacturedSolutionNavah_MS2<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionNavah_MS2<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionNavah_MS3<PHILIP_DIM,double>;
template class ManufacturedSolutionNavah_MS3<PHILIP_DIM,FadType>;
template class ManufacturedSolutionNavah_MS3<PHILIP_DIM,RadType>;
template class ManufacturedSolutionNavah_MS3<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionNavah_MS3<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionNavah_MS4<PHILIP_DIM,double>;
template class ManufacturedSolutionNavah_MS4<PHILIP_DIM,FadType>;
template class ManufacturedSolutionNavah_MS4<PHILIP_DIM,RadType>;
template class ManufacturedSolutionNavah_MS4<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionNavah_MS4<PHILIP_DIM,RadFadType>;
template class ManufacturedSolutionNavah_MS5<PHILIP_DIM,double>;
template class ManufacturedSolutionNavah_MS5<PHILIP_DIM,FadType>;
template class ManufacturedSolutionNavah_MS5<PHILIP_DIM,RadType>;
template class ManufacturedSolutionNavah_MS5<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionNavah_MS5<PHILIP_DIM,RadFadType>;

template class ManufacturedSolutionFactory<PHILIP_DIM,double>;
template class ManufacturedSolutionFactory<PHILIP_DIM,FadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM,RadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM,FadFadType>;
template class ManufacturedSolutionFactory<PHILIP_DIM,RadFadType>;
}
