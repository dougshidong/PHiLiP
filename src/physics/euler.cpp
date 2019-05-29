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
inline std::array<real,dim> Euler<dim,nstate,real>
::compute_velocities ( const std::array<real,nstate> &soln ) const
{
    std::array<real, dim> vel;
    const real density = soln[0];
    for (int idim=0; idim<dim; ++idim) {
        vel[idim] = soln[1+idim]/density;
    }
    return vel;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_pressure ( const std::array<real,nstate> &soln ) const
{
    const real density = soln[0];
    const real energy  = soln[1+dim];
    const std::array<real,dim> vel = compute_velocities(soln);
    real vel2 = 0;
    for (int d=0; d<dim; d++) {
        vel2 = vel2 + vel[d]*vel[d];
    }
    const real pressure = (gam-1.0)*(energy - 0.5*density*vel2);
    return pressure;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &soln ) const
{
    const real density = soln[0];
    const real pressure = compute_pressure(soln);
    const real sound = std::sqrt(pressure*gam/density);
    return sound;
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &soln,
    std::array<dealii::Tensor<1,dim,real>,nstate> &conv_flux) const
{
    const real density = soln[0];
    const real pressure = compute_pressure (soln);
    const std::array<real,dim> vel = compute_velocities(soln);
    const real tot_energy = soln[nstate-1];

    for (int fdim=0; fdim<dim; ++fdim) {
        // Density equation
        conv_flux[0][fdim] = soln[1+fdim];
        // Momentum equation
        for (int sdim=0; sdim<dim; ++sdim){
            conv_flux[1+sdim][fdim] = density*vel[fdim]*vel[sdim];
        }
        conv_flux[1+fdim][1+fdim] += pressure; // Add diagonal of pressure
        // Energy equation
        conv_flux[2+dim][fdim] = (soln[2+dim]+pressure)*vel[fdim];
    }
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*soln*/,
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
    const std::array<real,nstate> &/*soln*/,
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
    const std::array<real,nstate> &/*soln*/,
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

template class Euler < PHILIP_DIM, PHILIP_DIM+2, double >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, Sacado::Fad::DFad<double>  >;

} // Physics namespace
} // PHiLiP namespace


