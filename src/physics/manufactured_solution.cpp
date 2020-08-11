#include <Sacado.hpp>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>
#include <deal.II/base/function_time.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>

#include "manufactured_solution.h"

//#define ADDITIVE_SOLUTION
//#define COSINE_SOLUTION
#define EXP_SOLUTION
//#define ATAN_SOLUTION
//#define EVENPOLY_SOLUTION
//#define POLY_SOLUTION

template class dealii::FunctionTime<Sacado::Fad::DFad<double>>; // Needed by Function
template class dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>;

namespace PHiLiP {

bool isfinite(double value)
{
    return std::isfinite(static_cast<double>(value));
}

bool isfinite(Sacado::Fad::DFad<double> value)
{
    return std::isfinite(static_cast<double>(value.val()));
}

bool isfinite(Sacado::Fad::DFad<Sacado::Fad::DFad<double>> value)
{
    return std::isfinite(static_cast<double>(value.val().val()));
}
bool isfinite(Sacado::Rad::ADvar<Sacado::Fad::DFad<double>> value)
{
    return std::isfinite(static_cast<double>(value.val().val()));
}

template <int dim, typename real>
ManufacturedSolutionFunction<dim,real>
::ManufacturedSolutionFunction (const unsigned int nstate)
    : dealii::Function<dim,real>(nstate)
    , nstate(nstate)
    , base_values(nstate)
    , amplitudes(nstate)
    , frequencies(nstate)
{

    const double pi = atan(1)*4.0;
    //const double ee = exp(1);

    for (unsigned int s=0; s<nstate; s++) {
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
inline real ManufacturedSolutionFunction<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= sin( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }

#ifdef ADDITIVE_SOLUTION
    value = 0.0;
    for (int d=0; d<dim; d++) {
        value += amplitudes[istate]*sin( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
#endif

#ifdef COSINE_SOLUTION
    value = amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= cos( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
#endif

#ifdef EXP_SOLUTION
    value = 0.0;
    for (int d=0; d<dim; d++) {
        value += exp( point[d] );
        assert(isfinite(value));
    }
#endif

#ifdef EVENPOLY_SOLUTION
    value = 0.0;
    const double poly_max = 7;
    for (int d=0; d<dim; d++) {
        value += pow(point[d] + 0.5, poly_max);
    }
#endif

#ifdef POLY_SOLUTION
    value = 0.0;
    for (int d=0; d<dim; d++) {
        const real x = point[d];
        value += 1.0 + x - x*x - x*x*x + x*x*x*x - x*x*x*x*x + x*x*x*x*x*x + 0.001*sin(50*x);
    }
#endif

    value += base_values[istate];

#ifdef ATAN_SOLUTION
    value = 1.0;
	const real S1 = 50, S2 = -50;
	const real loc1 = 0.25, loc2 = 0.75;
    for (int d=0; d<dim; d++) {
		const real dimval = atan(S1*(point[d]-loc1)) + atan(S2*(point[d]-loc2));
        value *= dimval;
        assert(isfinite(value));
    }
#endif

    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    for (int dim_deri=0; dim_deri<dim; dim_deri++) {
        gradient[dim_deri] = amplitudes[istate] * frequencies[istate][dim_deri];
        for (int dim_trig=0; dim_trig<dim; dim_trig++) {
            const real angle = frequencies[istate][dim_trig] * point[dim_trig];
            if (dim_deri == dim_trig) gradient[dim_deri] *= cos( angle );
            if (dim_deri != dim_trig) gradient[dim_deri] *= sin( angle );
        }
        assert(isfinite(gradient[dim_deri]));
    }
    // Hard-coded is much more readable than the dimensionally generic one
    const real A = amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = frequencies[istate];
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

#ifdef ADDITIVE_SOLUTION
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
#endif

#ifdef COSINE_SOLUTION
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
#endif
#ifdef EXP_SOLUTION
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
#endif
#ifdef ATAN_SOLUTION
	const real S1 = 50, S2 = -50;
	const real loc1 = 0.25, loc2 = 0.75;
    if (dim==1) {
        gradient[0] = S1 / (std::pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (std::pow(S2*(point[0]-loc2), 2) + 1.0);
    }
    if (dim==2) {
		const real xval = atan(S1*(point[0]-loc1)) + atan(S2*(point[0]-loc2));
		const real yval = atan(S1*(point[1]-loc1)) + atan(S2*(point[1]-loc2));
        gradient[0] = S1 / (std::pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (std::pow(S2*(point[0]-loc2), 2) + 1.0);
		gradient[0] *= yval;
        gradient[1] = S1 / (std::pow(S1*(point[1]-loc1), 2) + 1.0);
        gradient[1] += S2 / (std::pow(S2*(point[1]-loc2), 2) + 1.0);
		gradient[1] *= xval;
    }
    if (dim==3) {
		const real xval = atan(S1*(point[0]-loc1)) + atan(S2*(point[0]-loc2));
		const real yval = atan(S1*(point[1]-loc1)) + atan(S2*(point[1]-loc2));
		const real zval = atan(S1*(point[2]-loc1)) + atan(S2*(point[2]-loc2));
        gradient[0] = S1 / (std::pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (std::pow(S2*(point[0]-loc2), 2) + 1.0);
		gradient[0] *= yval;
		gradient[0] *= zval;
        gradient[1] = S1 / (std::pow(S1*(point[1]-loc1), 2) + 1.0);
        gradient[1] += S2 / (std::pow(S2*(point[1]-loc2), 2) + 1.0);
		gradient[1] *= xval;
		gradient[1] *= zval;
        gradient[2] = S1 / (std::pow(S1*(point[2]-loc1), 2) + 1.0);
        gradient[2] += S2 / (std::pow(S2*(point[2]-loc2), 2) + 1.0);
		gradient[2] *= xval;
		gradient[2] *= yval;
    }
#endif
#ifdef EVENPOLY_SOLUTION
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
#endif
#ifdef POLY_SOLUTION
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
#endif
    return gradient;
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
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    // Hard-coded is much more readable than the dimensionally generic one
    const real A = amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = frequencies[istate];
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

#ifdef ADDITIVE_SOLUTION
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
#endif

#ifdef COSINE_SOLUTION
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
#endif
#ifdef EXP_SOLUTION
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
#endif
#ifdef ATAN_SOLUTION
    dealii::Tensor<1,dim,real> gradient;
	const real S1 = 50, S2 = -50;
	const real loc1 = 0.25, loc2 = 0.75;
    if (dim==1) {
        const real x = point[0];
        gradient[0] = S1 / (std::pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (std::pow(S2*(point[0]-loc2), 2) + 1.0);

        hessian[0][0] =
			-(2* std::pow(S1,3)*(-loc1 + x))/std::pow(1 + std::pow(S1*(-loc1 + x),2),2)
			-(2* std::pow(S2,3)*(-loc2 + x))/std::pow(1 + std::pow(S2*(-loc2 + x),2),2);
    }
    if (dim==2) {
        const real x = point[0], y = point[1];
		const real xval = atan(S1*(point[0]-loc1)) + atan(S2*(point[0]-loc2));
		const real yval = atan(S1*(point[1]-loc1)) + atan(S2*(point[1]-loc2));
        gradient[0] = S1 / (std::pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (std::pow(S2*(point[0]-loc2), 2) + 1.0);
		gradient[0] *= yval;
        gradient[1] = S1 / (std::pow(S1*(point[1]-loc1), 2) + 1.0);
        gradient[1] += S2 / (std::pow(S2*(point[1]-loc2), 2) + 1.0);
		gradient[1] *= xval;

        hessian[0][0] =
			-(2* std::pow(S1,3)*(-loc1 + x))/std::pow(1 + std::pow(S1*(-loc1 + x),2),2)
			-(2* std::pow(S2,3)*(-loc2 + x))/std::pow(1 + std::pow(S2*(-loc2 + x),2),2);
        hessian[0][0] *= yval;
        hessian[0][1] = (S1 / (std::pow(S1*(point[0]-loc1), 2) + 1.0) *  S2 / (std::pow(S2*(point[0]-loc2), 2) + 1.0));
        hessian[0][1] *= (S1 / (std::pow(S1*(point[1]-loc1), 2) + 1.0) *  S2 / (std::pow(S2*(point[1]-loc2), 2) + 1.0));
        hessian[1][0] = hessian[0][1];
        hessian[1][1] =
			-(2* std::pow(S1,3)*(-loc1 + y))/std::pow(1 + std::pow(S1*(-loc1 + y),2),2)
			-(2* std::pow(S2,3)*(-loc2 + y))/std::pow(1 + std::pow(S2*(-loc2 + y),2),2);

    }
    if (dim==3) {
		std::abort();
    }
#endif
#ifdef EVENPOLY_SOLUTION
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
#endif
#ifdef POLY_SOLUTION
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
#endif
    return hessian;
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

template class ManufacturedSolutionFunction<PHILIP_DIM,double>;
template class ManufacturedSolutionFunction<PHILIP_DIM,Sacado::Fad::DFad<double>>;
template class ManufacturedSolutionFunction<PHILIP_DIM,Sacado::Fad::DFad<Sacado::Fad::DFad<double>>>;
template class ManufacturedSolutionFunction<PHILIP_DIM,Sacado::Rad::ADvar<Sacado::Fad::DFad<double>>>;

}
