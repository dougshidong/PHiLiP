#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"


namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
real Euler<dim,nstate,real>
::compute_pressure ( const std::array<real,nstate> &solution )
{
    const real density = solution[0];
    const real energy  = solution[1+dim];
    real vel2 = 0.0; // velocity squared
    for (int d=0; d<dim; d++) {
        vel2 += std::pow(solution[1+d]/density, 2);
    }
    const real pressure = (gam-1.0)*(energy - 0.5*density*vel2);
    return pressure;
}

template <int dim, int nstate, typename real>
real Euler<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &solution )
{
    const real density = solution[0];
    const real energy  = solution[1+dim];
    real vel2 = 0.0; // velocity squared
    for (int d=0; d<dim; d++) {
        vel2 += std::pow(solution[1+d]/density, 2);
    }
    const real pressure = (gam-1.0)*(energy - 0.5*density*vel2);
    return pressure;
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &solution,
    std::array<dealii::Tensor<1,dim,real>,nstate> &conv_flux) const
{
    for (int i=0; i<nstate; ++i) {
        //conv_flux[i] = velocity_field * solution[i];
    }
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*solution*/,
    const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real,nstate> eig;
    for (int i=0; i<nstate; i++) {
        //eig[i] = advection_speed*normal;
    }
    return eig;
}


template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    std::array<dealii::Tensor<1,dim,real>,nstate> &diss_flux) const
{
    // No dissipation
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/,
    std::array<real,nstate> &source) const
{
    using phys = PhysicsBase<dim,nstate,real>;
    const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
    const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

    // int istate = 0;
    // if (dim==1) {
    //     const real x = pos[0];
    //     source[istate] = vel[0]*a*cos(a*x+d);
    // } else if (dim==2) {
    //     const real x = pos[0], y = pos[1];
    //     source[istate] = vel[0]*a*cos(a*x+d)*sin(b*y+e) +
    //                      vel[1]*b*sin(a*x+d)*cos(b*y+e);
    // } else if (dim==3) {
    //     const real x = pos[0], y = pos[1], z = pos[2];
    //     source[istate] =  vel[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
    //                       vel[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
    //                       vel[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f);
    // }

    // if (nstate > 1) {
    //     int istate = 1;
    //     if (dim==1) {
    //         const real x = pos[0];
    //         source[istate] = -vel[0]*a*sin(a*x+d);
    //     } else if (dim==2) {
    //         const real x = pos[0], y = pos[1];
    //         source[istate] = - vel[0]*a*sin(a*x+d)*cos(b*y+e)
    //                          - vel[1]*b*cos(a*x+d)*sin(b*y+e);
    //     } else if (dim==3) {
    //         const real x = pos[0], y = pos[1], z = pos[2];
    //         source[istate] =  - vel[0]*a*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
    //                           - vel[1]*b*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
    //                           - vel[2]*c*cos(a*x+d)*cos(b*y+e)*sin(c*z+f);
    //     }
    // }
}

// Instantiate explicitly

template class Euler < PHILIP_DIM, 3, double >;
template class Euler < PHILIP_DIM, 3, Sacado::Fad::DFad<double>  >;
template class Euler < PHILIP_DIM, 4, double >;
template class Euler < PHILIP_DIM, 4, Sacado::Fad::DFad<double>  >;
template class Euler < PHILIP_DIM, 5, double >;
template class Euler < PHILIP_DIM, 5, Sacado::Fad::DFad<double>  >;


} // Physics namespace
} // PHiLiP namespace


