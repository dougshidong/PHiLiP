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

    std::array<real,nstate> base_value;
    base_value[0] = pi/4.0;
    for (int i=0;i<dim;i++) {
        base_value[1+i] = ee/(i+1);
    }
    base_value[nstate-1] = ee/pi;



    std::array<dealii::Tensor<1,dim,real>,nstate> amplitudes;
    std::array<dealii::Tensor<1,dim,real>,nstate> frequencies;
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

    std::array<real,nstate> base_value;
    base_value[0] = pi/4.0;
    for (int i=0;i<dim;i++) {
        base_value[1+i] = ee/(i+1);
    }
    base_value[nstate-1] = ee/pi;


    std::array<dealii::Tensor<1,dim,real>,nstate> ampl;
    std::array<dealii::Tensor<1,dim,real>,nstate> freq;
    for (int s=0; s<nstate; s++) {
        for (int d=0; d<dim; d++) {
            ampl[s][d] = 0.5*base_value[s]*(dim-d)/dim*(nstate-s)/nstate;
            freq[s][d] = 1.0+sin(0.1+s*0.5+d*0.2);
        }
    }


    // Obtain primitive solution
    const std::array<real,nstate> conservative_manu_soln = manufactured_solution(pos);
    const std::array<real,nstate> primitive_soln = Euler<dim,nstate,real>::convert_conservative_to_primitive(conservative_manu_soln);
    const real density = primitive_soln[0];
    const std::array<real,dim> vel = Euler::extract_velocities_from_primitive(primitive_soln);
    const real pressure = primitive_soln[nstate-1];
    const real energy = conservative_manu_soln[nstate-1];

    // Derivatives of total energy w.r.t primitive variables
    const real v2 = Euler::compute_velocity_squared(vel);
    const real dedr = 0.5*v2;
    std::array<real,dim> dedv;
    for (int d=0;d<dim;d++) {
        dedv[d] = density*vel[d];
    }
    const real dedp = 1.0/(gam-1.0);

    std::array<real,nstate> source;
    source.fill(0.0);
    dealii::Table<3,real> dflux_dx(nstate, dim, dim); // state, flux_dim, deri_dim
    for (int deri_dim=0; deri_dim<dim; ++deri_dim){

        // Derivative of primitive solution w.r.t x, y, or z
        real drdx = ampl[0][deri_dim] * freq[0][deri_dim] * pi/2.0 * cos( freq[0][deri_dim] * pos[deri_dim] * pi / 2.0 );
        std::array<real,dim> dvdx;
        for (int vel_d=0;vel_d<dim;vel_d++) {
            dvdx[vel_d] = ampl[1+vel_d][deri_dim] * freq[1+vel_d][deri_dim] * pi/2.0 * cos( freq[1+vel_d][deri_dim] * pos[deri_dim] * pi / 2.0 );
        }
        real dpdx = ampl[nstate-1][deri_dim] * freq[nstate-1][deri_dim] * pi/2.0 * cos( freq[nstate-1][deri_dim] * pos[deri_dim] * pi / 2.0 );


        //for (int flux_dim=0; flux_dim<dim; ++flux_dim){
        // Only need the divergence of the flux, not all the derivatives
        const int flux_dim = deri_dim;
        {
            const real vflux = primitive_soln[1+flux_dim];
            const real dvflux_dx = dvdx[flux_dim];

            const real dmassflux_dx = drdx * vflux + dvflux_dx * density;

            // Mass flux
            dflux_dx[0][flux_dim][deri_dim] = dmassflux_dx;
            // Momentum flux
            for (int vel_d=0; vel_d<dim; ++vel_d) {
                dflux_dx[1+vel_d][flux_dim][deri_dim] = dvdx[vel_d] * density * vflux + dmassflux_dx*vel[vel_d];
            }
            dflux_dx[1+flux_dim][flux_dim][deri_dim] += dpdx;
            // Energy flux
            dflux_dx[nstate-1][flux_dim][deri_dim] = dvflux_dx * (energy + pressure) + dpdx * vflux;
            dflux_dx[nstate-1][flux_dim][deri_dim] += dedr * drdx * vflux;
            for (int vel_d=0; vel_d<dim; ++vel_d) {
                dflux_dx[nstate-1][flux_dim][deri_dim] += dedv[vel_d] * dvdx[vel_d] * vflux;
            }
            dflux_dx[nstate-1][flux_dim][deri_dim] += dedp * dpdx * vflux;
        }
        source[0] += dflux_dx[0][flux_dim][deri_dim];
        for (int vel_d=0; vel_d<dim; ++vel_d) {
            source[1+vel_d] += dflux_dx[1+vel_d][flux_dim][deri_dim];
        }
        source[nstate-1] += dflux_dx[nstate-1][flux_dim][deri_dim];
    }

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

    primitive_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        primitive_soln[1+d] = vel[d];
    }
    primitive_soln[nstate-1] = pressure;
    return primitive_soln;
}

template <int dim, int nstate, typename real>
inline std::array<real,nstate> Euler<dim,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const
{

    const real density = primitive_soln[0];
    const std::array<real,dim> velocities = extract_velocities_from_primitive(primitive_soln);

    std::array<real, nstate> conservative_soln;
    conservative_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        conservative_soln[1+d] = density*velocities[d];
    }
    conservative_soln[nstate-1] = compute_energy(primitive_soln);

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
    const real pressure = primitive_soln[nstate-1];
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
    const real energy  = conservative_soln[nstate-1];
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

    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_flux[1+velocity_dim][flux_dim] = density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += pressure; // Add diagonal of pressure
        // Energy equation
        conv_flux[nstate-1][flux_dim] = (tot_energy+pressure)*vel[flux_dim];
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
    (void) vel;
    (void) normal;
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


