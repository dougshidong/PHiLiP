#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "mhd.h"


namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::array<real,nstate> MHD<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/) const
{
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; s++) {
        source_term[s] = 0;
    }

    return source_term;
}

//incomplete
template <int dim, int nstate, typename real>
inline std::array<real,nstate> MHD<dim,nstate,real>
::convert_conservative_to_primitive ( const std::array<real,nstate> &conservative_soln ) const
{
    std::array<real, nstate> primitive_soln;

    real density = conservative_soln[0];
    dealii::Tensor<1,dim,real> vel = compute_velocities (conservative_soln);
    real pressure = compute_pressure (conservative_soln);

    primitive_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        primitive_soln[1+d] = vel[d];
    }
    primitive_soln[nstate-1] = pressure;
    return primitive_soln;
}

//incomplete
template <int dim, int nstate, typename real>
inline std::array<real,nstate> MHD<dim,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const
{

    const real density = primitive_soln[0];
    const dealii::Tensor<1,dim,real> velocities = extract_velocities_from_primitive(primitive_soln);

    std::array<real, nstate> conservative_soln;
    conservative_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        conservative_soln[1+d] = density*velocities[d];
    }
    conservative_soln[nstate-1] = compute_total_energy(primitive_soln);

    return conservative_soln;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> MHD<dim,nstate,real>
::compute_velocities ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    dealii::Tensor<1,dim,real> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }
    return vel;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_velocity_squared ( const dealii::Tensor<1,dim,real> &velocities ) const
{
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { vel2 = vel2 + velocities[d]*velocities[d]; }
    return vel2;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> MHD<dim,nstate,real>
::extract_velocities_from_primitive ( const std::array<real,nstate> &primitive_soln ) const
{
    dealii::Tensor<1,dim,real> velocities;
    for (int d=0; d<dim; d++) { velocities[d] = primitive_soln[1+d]; }
    return velocities;
}


//incomplete
template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const
{
    const real density = primitive_soln[0];
    const real pressure = primitive_soln[nstate-1];
    const dealii::Tensor<1,dim,real> velocities = extract_velocities_from_primitive(primitive_soln);
    const real vel2 = compute_velocity_squared(velocities);

    const real tot_energy = pressure / gamm1 + 0.5*density*vel2;
    return tot_energy;
}

//incomplete
template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_entropy_measure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    const real pressure = compute_pressure(conservative_soln);
    const real entropy_measure = pressure*pow(density,-gam);
    return entropy_measure;
}

//incomplete
template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_specific_enthalpy ( const std::array<real,nstate> &conservative_soln, const real pressure ) const
{
    const real density = conservative_soln[0];
    const real total_energy = conservative_soln[nstate-1];
    const real specific_enthalpy = (total_energy+pressure)/density;
    return specific_enthalpy;
}


template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_dimensional_temperature ( const std::array<real,nstate> &primitive_soln ) const
{
    const real density = primitive_soln[0];
    const real pressure = primitive_soln[nstate-1];
    const real temperature = gam*pressure/density;
    return temperature;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_temperature ( const std::array<real,nstate> &primitive_soln ) const
{
    const real dimensional_temperature = compute_dimensional_temperature(primitive_soln);
    const real temperature = dimensional_temperature /** mach_inf_sqr*/;
    return temperature;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_density_from_pressure_temperature ( const real pressure, const real temperature ) const
{
    const real density = gam*pressure/temperature /** mach_inf_sqr*/;
    return density;
}
template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_temperature_from_density_pressure ( const real density, const real pressure ) const
{
    const real temperature = gam*pressure/density /* mach_inf_sqr*/;
    return temperature;
}


template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    //std::cout << "density " << density << std::endl;

    const real tot_energy  = conservative_soln[nstate-1];
  //  std::cout << "tot_energy " << tot_energy << std::endl;

    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
   // std::cout << "vel1 " << vel[0] << std::endl
   //                      << vel[1] << std::endl
//                         << vel[2] <<std::endl;

    const real vel2 = compute_velocity_squared(vel);
    //std::cout << "vel ^2 " << vel2 <<std::endl;
    real pressure = gamm1*(tot_energy - 0.5*density*vel2);
    //std::cout << "calculated pressure is" << pressure << std::endl;
    if(pressure<0.0) {
        std::cout<<"Cannot compute pressure..."<<std::endl;
        std::cout<<"density "<<density<<std::endl;
        for(int d=0;d<dim;d++) std::cout<<"vel"<<d<<" "<<vel[d]<<std::endl;
        std::cout<<"energy "<<tot_energy<<std::endl;
    }
    assert(pressure>0.0);
    //if(pressure<1e-4) pressure = 0.01;
    return pressure;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    real density = conservative_soln[0];
    //if(density<1e-4) density = 0.01;
    if(density<0.0) {
        std::cout<<"density"<<density<<std::endl;
        std::abort();
    }
    assert(density > 0);
    const real pressure = compute_pressure(conservative_soln);
    //std::cout << "pressure is" << pressure << std::endl;
    const real sound = sqrt(pressure*gam/density);
    //std::cout << "sound is " << sound << std::endl;
    return sound;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_sound ( const real density, const real pressure ) const
{
    assert(density > 0);
    const real sound = sqrt(pressure*gam/density);
    return sound;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_mach_number ( const std::array<real,nstate> &conservative_soln ) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real velocity = sqrt(compute_velocity_squared(vel));
    const real sound = compute_sound (conservative_soln);
    const real mach_number = velocity/sound;
    return mach_number;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>
::compute_magnetic_energy (const std::array<real,nstate> &conservative_soln) const
{
    real magnetic_energy = 0;
    for (int i = 1; i <= 3; ++i)
        magnetic_energy += 1./2. * (conservative_soln[nstate - i] * conservative_soln[nstate - i] );
    return magnetic_energy;
}


// Split form functions:

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>::
compute_mean_density(const std::array<real,nstate> &soln_const,
                          const std::array<real,nstate> &soln_loop) const
{
    return (soln_const[0] + soln_loop[0])/2.;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>::
compute_mean_pressure(const std::array<real,nstate> &soln_const,
                      const std::array<real,nstate> &soln_loop) const
{
    real pressure_const = compute_pressure(soln_const);
    real pressure_loop = compute_pressure(soln_loop);
    return (pressure_const + pressure_loop)/2.;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> MHD<dim,nstate,real>::
compute_mean_velocities(const std::array<real,nstate> &soln_const,
                        const std::array<real,nstate> &soln_loop) const
{
    dealii::Tensor<1,dim,real> vel_const = compute_velocities(soln_const);
    dealii::Tensor<1,dim,real> vel_loop = compute_velocities(soln_loop);
    //return (vel_const + vel_loop)/2.;
    dealii::Tensor<1,dim,real> mean_vel;
    for (int d=0; d<0; ++d) {
        mean_vel[d] = (vel_const[d] + vel_loop[d]) * 0.5;
    }
    return mean_vel;
}

template <int dim, int nstate, typename real>
inline real MHD<dim,nstate,real>::
compute_mean_specific_energy(const std::array<real,nstate> &soln_const,
                             const std::array<real,nstate> &soln_loop) const
{
    return ((soln_const[nstate-1]/soln_const[0]) + (soln_loop[nstate-1]/soln_loop[0]))/2.;
}


template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MHD<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real density = conservative_soln[0];
    const real pressure = compute_pressure (conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real specific_total_energy = conservative_soln[nstate-1]/conservative_soln[0];
    const real specific_total_enthalpy = specific_total_energy + pressure/density;
    const real magnetic_energy = compute_magnetic_energy(conservative_soln);

    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_flux[1+velocity_dim][flux_dim] = density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += pressure + magnetic_energy; // Add diagonal of pressure and magnetic energy
        // Energy equation
        conv_flux[nstate-4][flux_dim] = density*vel[flux_dim]*specific_total_enthalpy;
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MHD<dim, nstate, real>
::convective_numerical_split_flux(const std::array<real,nstate> &soln_const,
                                  const std::array<real,nstate> &soln_loop) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    const real mean_density = compute_mean_density(soln_const, soln_loop);
    const real mean_pressure = compute_mean_pressure(soln_const, soln_loop);
    const dealii::Tensor<1,dim,real> mean_velocities = compute_mean_velocities(soln_const,soln_loop);
    const real mean_specific_energy = compute_mean_specific_energy(soln_const, soln_loop);

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * mean_velocities[flux_dim];//conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_velocities[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += mean_pressure; // Add diagonal of pressure
        // Energy equation
        conv_num_split_flux[nstate-1][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_specific_energy + mean_pressure * mean_velocities[flux_dim];
    }

    return conv_num_split_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MHD<dim, nstate, real>
::convective_surface_numerical_split_flux (
                const std::array< dealii::Tensor<1,dim,real>, nstate > &/*surface_flux*/,
                const std::array< dealii::Tensor<1,dim,real>, nstate > &flux_interp_to_surface) const
{
    return flux_interp_to_surface;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> MHD<dim,nstate,real>
::convective_normal_flux (const std::array<real,nstate> &conservative_soln, const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real, nstate> conv_normal_flux;
    const real density = conservative_soln[0];
    const real pressure = compute_pressure (conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    //const real normal_vel = vel*normal;
    real normal_vel = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel += vel[d]*normal[d];
    }
    const real total_energy = conservative_soln[nstate-1];
    const real specific_total_enthalpy = (total_energy + pressure) / density;

    const real rhoV = density*normal_vel;
    // Density equation
    conv_normal_flux[0] = rhoV;
    // Momentum equation
    for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
        conv_normal_flux[1+velocity_dim] = rhoV*vel[velocity_dim] + normal[velocity_dim] * pressure;
    }
    // Energy equation
    conv_normal_flux[nstate-1] = rhoV*specific_total_enthalpy;
    return conv_normal_flux;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> MHD<dim,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // See Blazek Appendix A.9 p. 429-430
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel_normal = 0.0;
    for (int d=0;d<dim;d++) { vel_normal += vel[d] * normal[d]; }

    const real vel2 = compute_velocity_squared(vel);
    const real phi = 0.5*gamm1 * vel2;

    const real density = conservative_soln[0];
    const real tot_energy = conservative_soln[nstate-1];
    const real E = tot_energy / density;
    const real a1 = gam*E-phi;
    const real a2 = gam-1.0;
    const real a3 = gam-2.0;

    dealii::Tensor<2,nstate,real> jacobian;
    for (int d=0; d<dim; ++d) {
        jacobian[0][1+d] = normal[d];
    }
    for (int row_dim=0; row_dim<dim; ++row_dim) {
        jacobian[1+row_dim][0] = normal[row_dim]*phi - vel[row_dim] * vel_normal;
        for (int col_dim=0; col_dim<dim; ++col_dim){
            if (row_dim == col_dim) {
                jacobian[1+row_dim][1+col_dim] = vel_normal - a3*normal[row_dim]*vel[row_dim];
            } else {
                jacobian[1+row_dim][1+col_dim] = normal[col_dim]*vel[row_dim] - a2*normal[row_dim]*vel[col_dim];
            }
        }
        jacobian[1+row_dim][nstate-1] = normal[row_dim]*a2;
    }
    jacobian[nstate-1][0] = vel_normal*(phi-a1);
    for (int d=0; d<dim; ++d){
        jacobian[nstate-1][1+d] = normal[d]*a1 - a2*vel[d]*vel_normal;
    }
    jacobian[nstate-1][nstate-1] = gam*vel_normal;

    return jacobian;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> MHD<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    std::array<real,nstate> eig;
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    for (int i=0; i<nstate; i++) {
        eig[i] = vel_dot_n;
        //eig[i] = advection_speed*normal;

        //eig[i] = 1.0;
        //eig[i] = -1.0;
    }
    return eig;
}
template <int dim, int nstate, typename real>
real MHD<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    //std::cout << "going to calculate max eig" << std::endl;
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    //std::cout << "velocities calculated" << std::endl;

    const real sound = compute_sound (conservative_soln);
    //std::cout << "sound calculated" << std::endl;

    /*const*/ real vel2 = compute_velocity_squared(vel);
    //std::cout << "vel2 calculated" << std::endl;

    if (vel2 < 0.0001)
        vel2 = 0.0001;

    const real max_eig = sqrt(vel2) + sound;
    //std::cout << "max eig calculated" << std::endl;

    return max_eig;
}


template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MHD<dim,nstate,real>
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

//template <int dim, int nstate, typename real>
//void MHD<dim,nstate,real>
//::boundary_face_values (
//   const int boundary_type,
//   const dealii::Point<dim, real> &pos,
//   const dealii::Tensor<1,dim,real> &normal_int,
//   const std::array<real,nstate> &soln_int,
//   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
//   std::array<real,nstate> &soln_bc,
//   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
//{
//    // NEED TO PROVIDE AS INPUT **************************************
//    const real total_inlet_pressure = pressure_inf*pow(1.0+0.5*gamm1*mach_inf_sqr, gam/gamm1);
//    const real total_inlet_temperature = temperature_inf*pow(total_inlet_pressure/pressure_inf, gamm1/gam);
//
//    if (boundary_type == 1000) {
//        // Manufactured solution
//        std::array<real,nstate> conservative_boundary_values;
//        std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
//        for (int s=0; s<nstate; s++) {
//            conservative_boundary_values[s] = this->manufactured_solution_function.value (pos, s);
//            boundary_gradients[s] = this->manufactured_solution_function.gradient (pos, s);
//        }
//        std::array<real,nstate> primitive_boundary_values = convert_conservative_to_primitive(conservative_boundary_values);
//        for (int istate=0; istate<nstate; ++istate) {
//
//            std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(conservative_boundary_values, normal_int);
//            const bool inflow = (characteristic_dot_n[istate] <= 0.);
//
//            if (inflow) { // Dirichlet boundary condition
//
//                soln_bc[istate] = conservative_boundary_values[istate];
//                soln_grad_bc[istate] = soln_grad_int[istate];
//
//                // Only set the pressure and velocity
//                // primitive_boundary_values[0] = soln_int[0];;
//                // for(int d=0;d<dim;d++){
//                //    primitive_boundary_values[1+d] = soln_int[1+d]/soln_int[0];;
//                //}
//                conservative_boundary_values = convert_primitive_to_conservative(primitive_boundary_values);
//                //conservative_boundary_values[nstate-1] = soln_int[nstate-1];
//                soln_bc[istate] = conservative_boundary_values[istate];
//
//            } else { // Neumann boundary condition
//                // soln_bc[istate] = -soln_int[istate]+2*conservative_boundary_values[istate];
//                soln_bc[istate] = soln_int[istate];
//
//                // **************************************************************************************************************
//                // Note I don't know how to properly impose the soln_grad_bc to obtain an adjoint consistent scheme
//                // Currently, Neumann boundary conditions are only imposed for the linear advection
//                // Therefore, soln_grad_bc does not affect the solution
//                // **************************************************************************************************************
//                soln_grad_bc[istate] = soln_grad_int[istate];
//                //soln_grad_bc[istate] = boundary_gradients[istate];
//                //soln_grad_bc[istate] = -soln_grad_int[istate]+2*boundary_gradients[istate];
//            }
//
//            // HARDCODE DIRICHLET BC
//            soln_bc[istate] = conservative_boundary_values[istate];
//
//        }
//    } else if (boundary_type == 1001) {
//        // No penetration,
//        // Given by Algorithm II of the following paper
//        // Krivodonova, L., and Berger, M.,
//        // “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
//        // Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
//        const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(soln_int);
//
//        // Copy density and pressure
//        std::array<real,nstate> primitive_boundary_values;
//        primitive_boundary_values[0] = primitive_interior_values[0];
//        primitive_boundary_values[nstate-1] = primitive_interior_values[nstate-1];
//
//        const dealii::Tensor<1,dim,real> surface_normal = -normal_int;
//        const dealii::Tensor<1,dim,real> velocities_int = extract_velocities_from_primitive(primitive_interior_values);
//        const dealii::Tensor<1,dim,real> velocities_bc = velocities_int - 2.0*(velocities_int*surface_normal)*surface_normal;
//        for (int d=0; d<dim; ++d) {
//            primitive_boundary_values[1+d] = velocities_bc[d];
//        }
//
//        soln_bc = convert_primitive_to_conservative(primitive_boundary_values);
//
//    } else if (boundary_type == 1002) {
//        // Pressure Outflow Boundary Condition (back pressure)
//        // Carlson 2011, sec. 2.4
//
//        const real back_pressure = 0.99; // Make it as an input later on
//
//        const real mach_int = compute_mach_number(soln_int);
//        const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(soln_int);
//        const real pressure_int = primitive_interior_values[nstate-1];
//
//        const real radicant = 1.0+0.5*gamm1*mach_inf_sqr;
//        const real pressure_inlet = total_inlet_pressure * pow(radicant, -gam/gamm1);
//        const real pressure_bc = (mach_int >= 1) ? pressure_int : back_pressure*pressure_inlet;
//        const real temperature_int = compute_temperature(primitive_interior_values);
//
//        // Assign primitive boundary values
//        std::array<real,nstate> primitive_boundary_values;
//        primitive_boundary_values[0] = compute_density_from_pressure_temperature(pressure_bc, temperature_int);
//        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = primitive_interior_values[1+d]; }
//        primitive_boundary_values[nstate-1] = pressure_bc;
//
//        soln_bc = convert_primitive_to_conservative(primitive_boundary_values);
//
//        // Supersonic, simply extrapolate
//        if (mach_int > 1.0) {
//            soln_bc = soln_int;
//        }
//
//    } else if (boundary_type == 1003) {
//        // Inflow
//        // Carlson 2011, sec. 2.2 & sec 2.9
//
//        const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(soln_int);
//
//        const dealii::Tensor<1,dim,real> normal = -normal_int;
//
//        const real                       density_i    = primitive_interior_values[0];
//        const dealii::Tensor<1,dim,real> velocities_i = extract_velocities_from_primitive(primitive_interior_values);
//        const real                       pressure_i   = primitive_interior_values[nstate-1];
//
//        const real                       normal_vel_i = velocities_i*normal;
//        const real                       sound_i      = compute_sound(soln_int);
//        //const real                       mach_i       = std::abs(normal_vel_i)/sound_i;
//
//        //const dealii::Tensor<1,dim,real> velocities_o = velocities_inf;
//        //const real                       normal_vel_o = velocities_o*normal;
//        //const real                       sound_o      = sound_inf;
//        //const real                       mach_o       = mach_inf;
//
//        if(mach_inf < 1.0) {
//            //std::cout << "Subsonic inflow, mach=" << mach_i << std::endl;
//            // Subsonic inflow, sec 2.7
//
//            // Want to solve for c_b (sound_bc), to then solve for U (velocity_magnitude_bc) and M_b (mach_bc)
//            // Eq. 37
//            const real riemann_pos = normal_vel_i + 2.0*sound_i/gamm1;
//            // Could evaluate enthalpy from primitive like eq.36, but easier to use the following
//            const real specific_total_energy = soln_int[nstate-1]/density_i;
//            const real specific_total_enthalpy = specific_total_energy + pressure_i/density_i;
//            // Eq. 43
//            const real a = 1.0+2.0/gamm1;
//            const real b = -2.0*riemann_pos;
//            const real c = 0.5*gamm1 * (riemann_pos*riemann_pos - 2.0*specific_total_enthalpy);
//            // Eq. 42
//            const real term1 = -0.5*b/a;
//            const real term2= 0.5*sqrt(b*b-4.0*a*c)/a;
//            const real sound_bc1 = term1 + term2;
//            const real sound_bc2 = term1 - term2;
//            // Eq. 44
//            const real sound_bc  = std::max(sound_bc1, sound_bc2);
//            // Eq. 45
//            //const real velocity_magnitude_bc = 2.0*sound_bc/gamm1 - riemann_pos;
//            const real velocity_magnitude_bc = riemann_pos - 2.0*sound_bc/gamm1;
//            const real mach_bc = velocity_magnitude_bc/sound_bc;
//            // Eq. 46
//            const real radicant = 1.0+0.5*gamm1*mach_bc*mach_bc;
//            const real pressure_bc = total_inlet_pressure * pow(radicant, -gam/gamm1);
//            const real temperature_bc = total_inlet_temperature * pow(radicant, -1.0);
//            //std::cout << " pressure_bc " << pressure_bc << "pressure_inf" << pressure_inf << std::endl;
//            //std::cout << " temperature_bc " << temperature_bc << "temperature_inf" << temperature_inf << std::endl;
//            //
//
//            const real density_bc  = compute_density_from_pressure_temperature(pressure_bc, temperature_bc);
//            std::array<real,nstate> primitive_boundary_values;
//            primitive_boundary_values[0] = density_bc;
//            for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = velocity_magnitude_bc*normal[d]; }
//            primitive_boundary_values[nstate-1] = pressure_bc;
//            soln_bc = convert_primitive_to_conservative(primitive_boundary_values);
//
//            //std::cout << " entropy_bc " << compute_entropy_measure(soln_bc) << "entropy_inf" << entropy_inf << std::endl;
//
//        } else {
//            // Supersonic inflow, sec 2.9
//            // Specify all quantities through
//            // total_inlet_pressure, total_inlet_temperature, mach_inf & angle_of_attack
//            //std::cout << "Supersonic inflow, mach=" << mach_i << std::endl;
//            const real radicant = 1.0+0.5*gamm1*mach_inf_sqr;
//            const real static_inlet_pressure    = total_inlet_pressure * pow(radicant, -gam/gamm1);
//            const real static_inlet_temperature = total_inlet_temperature * pow(radicant, -1.0);
//
//            const real pressure_bc = static_inlet_pressure;
//            const real temperature_bc = static_inlet_temperature;
//            const real density_bc  = compute_density_from_pressure_temperature(pressure_bc, temperature_bc);
//            const real sound_bc = sqrt(gam * pressure_bc / density_bc);
//            const real velocity_magnitude_bc = mach_inf * sound_bc;
//
//            // Assign primitive boundary values
//            std::array<real,nstate> primitive_boundary_values;
//            primitive_boundary_values[0] = density_bc;
//            for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = -velocity_magnitude_bc*normal_int[d]; } // minus since it's inflow
//            primitive_boundary_values[nstate-1] = pressure_bc;
//            soln_bc = convert_primitive_to_conservative(primitive_boundary_values);
//            //std::cout << "Inlet density : " << density_bc << std::endl;
//            //std::cout << "Inlet vel_x   : " << primitive_boundary_values[1] << std::endl;
//            //std::cout << "Inlet vel_y   : " << primitive_boundary_values[2] << std::endl;
//            //std::cout << "Inlet pressure: " << pressure_bc << std::endl;
//        }
//
//    } else if (boundary_type == 1004) {
//        // Farfield boundary condition
//        const real density_bc = density_inf;
//        const real pressure_bc = 1.0/(gam*mach_inf_sqr);
//        std::array<real,nstate> primitive_boundary_values;
//        primitive_boundary_values[0] = density_bc;
//        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = velocities_inf[d]; } // minus since it's inflow
//        primitive_boundary_values[nstate-1] = pressure_bc;
//        soln_bc = convert_primitive_to_conservative(primitive_boundary_values);
//        //std::cout << "Density inf " << soln_bc[0] << std::endl;
//        //std::cout << "momxinf " << soln_bc[1] << std::endl;
//        //std::cout << "momxinf " << soln_bc[2] << std::endl;
//        //std::cout << "energy inf " << soln_bc[3] << std::endl;
//    } else{
//        std::cout << "Invalid boundary_type: " << boundary_type << std::endl;
//        std::abort();
//    }
//}
//
//template <int dim, int nstate, typename real>
//dealii::Vector<double> MHD<dim,nstate,real>::post_compute_derived_quantities_vector (
//    const dealii::Vector<double>              &uh,
//    const std::vector<dealii::Tensor<1,dim> > &duh,
//    const std::vector<dealii::Tensor<2,dim> > &dduh,
//    const dealii::Tensor<1,dim>               &normals,
//    const dealii::Point<dim>                  &evaluation_points) const
//{
//    std::vector<std::string> names = post_get_names ();
//    dealii::Vector<double> computed_quantities = PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
//    unsigned int current_data_index = computed_quantities.size() - 1;
//    computed_quantities.grow_or_shrink(names.size());
//    if constexpr (std::is_same<real,double>::value) {
//
//        std::array<double, nstate> conservative_soln;
//        for (unsigned int s=0; s<nstate; ++s) {
//            conservative_soln[s] = uh(s);
//        }
//        const std::array<double, nstate> primitive_soln = convert_conservative_to_primitive(conservative_soln);
//
//        // Density
//        computed_quantities(++current_data_index) = primitive_soln[0];
//        // Velocities
//        for (unsigned int d=0; d<dim; ++d) {
//            computed_quantities(++current_data_index) = primitive_soln[1+d];
//        }
//        // Momentum
//        for (unsigned int d=0; d<dim; ++d) {
//            computed_quantities(++current_data_index) = conservative_soln[1+d];
//        }
//        // Energy
//        computed_quantities(++current_data_index) = conservative_soln[nstate-1];
//        // Pressure
//        computed_quantities(++current_data_index) = primitive_soln[nstate-1];
//        // Pressure
//        computed_quantities(++current_data_index) = compute_temperature(primitive_soln);
//        // Entropy generation
//        computed_quantities(++current_data_index) = compute_entropy_measure(conservative_soln) - entropy_inf;
//        // Mach Number
//        computed_quantities(++current_data_index) = compute_mach_number(conservative_soln);
//
//    }
//    if (computed_quantities.size()-1 != current_data_index) {
//        std::cout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
//                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
//                  << std::endl;
//    }
//
//    return computed_quantities;
//}
//
//template <int dim, int nstate, typename real>
//std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> MHD<dim,nstate,real>
//::post_get_data_component_interpretation () const
//{
//    namespace DCI = dealii::DataComponentInterpretation;
//    std::vector<DCI::DataComponentInterpretation> interpretation = PhysicsBase<dim,nstate,real>::post_get_data_component_interpretation (); // state variables
//    interpretation.push_back (DCI::component_is_scalar); // Density
//    for (unsigned int d=0; d<dim; ++d) {
//        interpretation.push_back (DCI::component_is_part_of_vector); // Velocity
//    }
//    for (unsigned int d=0; d<dim; ++d) {
//        interpretation.push_back (DCI::component_is_part_of_vector); // Momentum
//    }
//    interpretation.push_back (DCI::component_is_scalar); // Energy
//    interpretation.push_back (DCI::component_is_scalar); // Pressure
//    interpretation.push_back (DCI::component_is_scalar); // Temperature
//    interpretation.push_back (DCI::component_is_scalar); // Entropy generation
//    interpretation.push_back (DCI::component_is_scalar); // Mach number
//
//    std::vector<std::string> names = post_get_names();
//    if (names.size() != interpretation.size()) {
//        std::cout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
//    }
//    return interpretation;
//}
//
//
//template <int dim, int nstate, typename real>
//std::vector<std::string> MHD<dim,nstate,real> ::post_get_names () const
//{
//    std::vector<std::string> names = PhysicsBase<dim,nstate,real>::post_get_names ();
//    names.push_back ("density");
//    for (unsigned int d=0; d<dim; ++d) {
//      names.push_back ("velocity");
//    }
//    for (unsigned int d=0; d<dim; ++d) {
//      names.push_back ("momentum");
//    }
//    names.push_back ("energy");
//    names.push_back ("pressure");
//    names.push_back ("temperature");
//
//    names.push_back ("entropy_generation");
//    names.push_back ("mach_number");
//    return names;
//}
//
//template <int dim, int nstate, typename real>
//dealii::UpdateFlags MHD<dim,nstate,real>
//::post_get_needed_update_flags () const
//{
//    //return update_values | update_gradients;
//    return dealii::update_values;
//}

// Instantiate explicitly
template class MHD < PHILIP_DIM, 8, double >;
template class MHD < PHILIP_DIM, 8, FadType >;
template class MHD < PHILIP_DIM, 8, RadType >;
template class MHD < PHILIP_DIM, 8, FadFadType >;
template class MHD < PHILIP_DIM, 8, RadFadType >;

} // Physics namespace
} // PHiLiP namespace
