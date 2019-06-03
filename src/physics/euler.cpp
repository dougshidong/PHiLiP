#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/base/table.h>

#include "physics.h"


namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::manufactured_solution (const dealii::Point<dim,double> &pos) const
{
    const double pi = atan(1)*4.0;
    const double ee = exp(1);

    dealii::Table<1,real> base_value(nstate);
    base_value[0] = pi/4;
    for (int i=0;i<dim;i++) {
        base_value[1+i] = ee/(i+1);
    }
    base_value[nstate-1] = ee/pi;


    dealii::Table<2,real> amplitudes(nstate, dim);
    dealii::Table<2,real> frequencies(nstate, dim);
    for (int s=0; s<nstate; s++) {
        for (int d=0; d<dim; d++) {
            amplitudes[s][d] = 0.5*base_value[s]*(dim-d)/dim*(nstate-s)/nstate;
            frequencies[s][d] = 1.0+sin(0.1+s*0.5+d*0.2);
        }
    }

    // Density, velocity_x, velocity_y, velocity_z, pressure
    std::array<real,nstate> primitive_soln;
    for (int s=0; s<nstate; s++) {
        primitive_soln[s] = base_value[s];
        for (int d=0; d<dim; d++) {
            primitive_soln[s] += amplitudes[s][d] * sin( frequencies[s][d] * pos[d] * pi / 2.0 );
        }
    }

    return convert_primitive_to_conservative(primitive_soln);
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    const double pi = atan(1)*4.0;
    const double ee = exp(1);

    dealii::Table<1,real> base_value(nstate);
    base_value[0] = pi/4;
    for (int i=0;i<dim;i++) {
        base_value[1+i] = ee/(i+1);
    }
    base_value[nstate-1] = ee/pi;

    dealii::Table<2,real> amplitudes(nstate, dim);
    dealii::Table<2,real> frequencies(nstate, dim);
    for (int s=0; s<nstate; s++) {
        for (int d=0; d<nstate; d++) {
            amplitudes[s][d] = 0.5*base_value[s]*(dim-d)/dim*(nstate-s)/nstate;
            frequencies[s][d] = 1.0+sin(0.1+s*0.5+d*0.2);
        }
    }

    std::array<real,nstate> source;

    return source;
}

template <int dim, int nstate, typename real>
inline std::array<real,nstate> Euler<dim,nstate,real>
::convert_conservative_to_primitive ( const std::array<real,nstate> &conservative_soln ) const
{
    std::array<real, nstate> primitive_soln;

    real density = conservative_soln[0];
    std::array<real, dim> vel = compute_velocities (conservative_soln);
    real pressure = compute_pressure (conservative_soln);

    primitive_soln[0] = conservative_soln[0];
    for (int d=0; d<dim; ++d) {
        primitive_soln[1+d] = vel[d];
    }
    primitive_soln[1+dim] = pressure;
    return primitive_soln;
}

template <int dim, int nstate, typename real>
inline std::array<real,nstate> Euler<dim,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const
{

    const real density = primitive_soln[0];
    const std::array<real,dim> velocities = extract_velocities_from_primitive(primitive_soln);
    const real pressure = primitive_soln[1+dim];

    std::array<real, nstate> conservative_soln;
    conservative_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        conservative_soln[1+d] = density*velocities[d];
    }
    conservative_soln[1+dim] = compute_energy(primitive_soln);

    return conservative_soln;
}

template <int dim, int nstate, typename real>
inline std::array<real,dim> Euler<dim,nstate,real>
::compute_velocities ( const std::array<real,nstate> &conservative_soln ) const
{
    std::array<real, dim> vel;
    const real density = conservative_soln[0];
    for (int d=0; d<dim; ++d) {
        vel[d] = conservative_soln[1+d]/density;
    }
    return vel;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_velocity_squared ( const std::array<real,dim> &velocities ) const
{
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { vel2 = vel2 + velocities[d]*velocities[d]; }
    return vel2;
}

template <int dim, int nstate, typename real>
inline std::array<real,dim> Euler<dim,nstate,real>
::extract_velocities_from_primitive ( const std::array<real,nstate> &primitive_soln ) const
{
    std::array<real,dim> velocities;
    for (int d=0; d<dim; d++) { velocities[d] = primitive_soln[1+d]; }
    return velocities;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_energy ( const std::array<real,nstate> &primitive_soln ) const
{
    const real density = primitive_soln[0];
    const real pressure = primitive_soln[1+dim];
    const std::array<real,dim> velocities = extract_velocities_from_primitive(primitive_soln);
    const real vel2 = compute_velocity_squared(velocities);

    const real energy = pressure / (gam-1.0) + 0.5*density*vel2;
    return energy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    const real energy  = conservative_soln[1+dim];
    const std::array<real,dim> vel = compute_velocities(conservative_soln);
    const real vel2 = compute_velocity_squared(vel);
    const real pressure = (gam-1.0)*(energy - 0.5*density*vel2);
    return pressure;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    const real pressure = compute_pressure(conservative_soln);
    const real sound = std::sqrt(pressure*gam/density);
    return sound;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real density = conservative_soln[0];
    const real pressure = compute_pressure (conservative_soln);
    const std::array<real,dim> vel = compute_velocities(conservative_soln);
    const real tot_energy = conservative_soln[nstate-1];

    for (int fdim=0; fdim<dim; ++fdim) {
        // Density equation
        conv_flux[0][fdim] = conservative_soln[1+fdim];
        // Momentum equation
        for (int sdim=0; sdim<dim; ++sdim){
            conv_flux[1+sdim][fdim] = density*vel[fdim]*vel[sdim];
        }
        conv_flux[1+fdim][1+fdim] += pressure; // Add diagonal of pressure
        // Energy equation
        conv_flux[2+dim][fdim] = (conservative_soln[2+dim]+pressure)*vel[fdim];
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const std::array<real,dim> vel = compute_velocities(conservative_soln);
    std::array<real,nstate> eig;
    for (int i=0; i<nstate; i++) {
        //eig[i] = advection_speed*normal;
    }
    return eig;
}
template <int dim, int nstate, typename real>
real Euler<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    const std::array<real,dim> vel = compute_velocities(conservative_soln);
    const real sound = compute_sound (conservative_soln);
    real speed = 0.0;
    for (int i=0; i<dim; i++) {
        speed = speed + vel[i]*vel[i];
    }
    const real max_eig = sqrt(speed) + sound;
    return max_eig;
}


template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipation
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

// Instantiate explicitly

template class Euler < PHILIP_DIM, PHILIP_DIM+2, double >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, Sacado::Fad::DFad<double>  >;

} // Physics namespace
} // PHiLiP namespace


