#include <Sacado.hpp>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>
#include <deal.II/base/function_time.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>

#include "manufactured_solution.h"

template class dealii::FunctionTime<Sacado::Fad::DFad<double>>; // Needed by Function
template class dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>;

namespace PHiLiP {

bool isfinite(Sacado::Fad::DFad<double> value)
{
    return std::isfinite(static_cast<double>(value.val()));
}

template <int dim, typename real>
ManufacturedSolutionFunction<dim,real>
::ManufacturedSolutionFunction (const unsigned int nstate)
    :
    dealii::Function<dim,real>(nstate)
    , base_values(nstate)
    , amplitudes(nstate)
    , frequencies(nstate)
{

    const double pi = atan(1)*4.0;
    //const double ee = exp(1);

    for (int s=0; s<(int)nstate; s++) {
        base_values[s] = (s+1.0)/nstate;
        base_values[nstate-1] = 10;
        amplitudes[s] = 0.2*base_values[s]*sin((static_cast<double>(nstate)-s)/nstate);
        //amplitudes[s] = 0.1;
        //std::cout<< s << " AMPLITUDES[S] "<< amplitudes[s] << std::endl;
        for (int d=0; d<dim; d++) {
            frequencies[s][d] = 3.0 + sin(0.1+s*0.5+d*0.2) *  pi / 2.0;
            //frequencies[s][d] = 1.0;
            //frequencies[s][d] = sin(0.1+s*0.5)*(d+1.0)/dim;
            //std::cout<< d << "FREQUENCIES[S][D] "<< frequencies[s][d] << std::endl;
        }
    }
}

template <int dim, typename real>
inline real ManufacturedSolutionFunction<dim,real>
::value (const dealii::Point<dim> &point, const unsigned int istate) const
{
    real value = amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= sin( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,real>
::gradient (const dealii::Point<dim> &point, const unsigned int istate) const
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
    return gradient;
}
template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,real>
::gradient_fd (const dealii::Point<dim> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    const double eps=1e-6;
    for (int dim_deri=0; dim_deri<dim; dim_deri++) {
        dealii::Point<dim> pert_p = point;
        dealii::Point<dim> pert_m = point;
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
::hessian (const dealii::Point<dim> &point, const unsigned int istate) const
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
    return hessian;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionFunction<dim,real>
::hessian_fd (const dealii::Point<dim> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    const double eps=1e-4;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=d1; d2<dim; d2++) {
            dealii::Point<dim> pert_p_p = point;
            dealii::Point<dim> pert_p_m = point;
            dealii::Point<dim> pert_m_p = point;
            dealii::Point<dim> pert_m_m = point;

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
inline std::vector<real> ManufacturedSolutionFunction<dim,real>
::stdvector_values (const dealii::Point<dim> &point) const
{
    const int nstate = this->n_components;
    std::vector<real> values(nstate);
    for (int s=0; s<nstate; s++) { values[s] = value(point, s); }
    return values;
}

template class ManufacturedSolutionFunction<PHILIP_DIM,double>;
template class ManufacturedSolutionFunction<PHILIP_DIM,Sacado::Fad::DFad<double>>;

}
