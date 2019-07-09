#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "euler.h"


namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    std::array<real,nstate> manufactured_solution;
    for (int s=0; s<nstate; s++) {
        manufactured_solution[s] = this->manufactured_solution_function.value (pos, s);
    }
    std::vector<dealii::Tensor<1,dim,real>> manufactured_solution_gradient_dealii(nstate);
    this->manufactured_solution_function.vector_gradient (pos, manufactured_solution_gradient_dealii);
    std::array<dealii::Tensor<1,nstate,real>,dim> manufactured_solution_gradient;
    for (int d=0;d<dim;d++) {
        for (int s=0; s<nstate; s++) {
            manufactured_solution_gradient[d][s] = manufactured_solution_gradient_dealii[s][d];
        }
    }

    dealii::Tensor<1,nstate,real> convective_flux_divergence;
    for (int d=0;d<dim;d++) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = convective_flux_directional_jacobian(manufactured_solution, normal);
        convective_flux_divergence += jacobian*manufactured_solution_gradient[d];
    }
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; s++) {
        source_term[s] = convective_flux_divergence[s];
    }

    return source_term;
}

template <int dim, int nstate, typename real>
inline std::array<real,nstate> Euler<dim,nstate,real>
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

template <int dim, int nstate, typename real>
inline std::array<real,nstate> Euler<dim,nstate,real>
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
inline dealii::Tensor<1,dim,real> Euler<dim,nstate,real>
::compute_velocities ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    dealii::Tensor<1,dim,real> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }
    return vel;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_velocity_squared ( const dealii::Tensor<1,dim,real> &velocities ) const
{
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { vel2 = vel2 + velocities[d]*velocities[d]; }
    return vel2;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Euler<dim,nstate,real>
::extract_velocities_from_primitive ( const std::array<real,nstate> &primitive_soln ) const
{
    dealii::Tensor<1,dim,real> velocities;
    for (int d=0; d<dim; d++) { velocities[d] = primitive_soln[1+d]; }
    return velocities;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const
{
    const real density = primitive_soln[0];
    const real pressure = primitive_soln[nstate-1];
    const dealii::Tensor<1,dim,real> velocities = extract_velocities_from_primitive(primitive_soln);
    const real vel2 = compute_velocity_squared(velocities);

    const real tot_energy = pressure / (gam-1.0) + 0.5*density*vel2;
    return tot_energy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_entropy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real entropy = 0.0;
    return entropy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_dimensional_temperature ( const std::array<real,nstate> &primitive_soln ) const
{
    const real density = primitive_soln[0];
    const real pressure = primitive_soln[nstate-1];
    const real temperature = gam*pressure/density;
    return temperature;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_temperature ( const std::array<real,nstate> &primitive_soln ) const
{
    const real dimensional_temperature = compute_dimensional_temperature(primitive_soln);
    const real temperature = dimensional_temperature * mach_inf*mach_inf;
    return temperature;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    const real tot_energy  = conservative_soln[nstate-1];
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real vel2 = compute_velocity_squared(vel);
    real pressure = (gam-1.0)*(tot_energy - 0.5*density*vel2);
    if(pressure<1e-4) {
        std::cout<<"density"<<density<<std::endl;
        for(int d=0;d<dim;d++) std::cout<<"vel"<<d<<" "<<vel[d]<<std::endl;
        std::cout<<"energy"<<tot_energy<<std::endl;
    }
    assert(pressure>0.0);
    //if(pressure<1e-4) pressure = 0.01;
    return pressure;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    real density = conservative_soln[0];
    assert(density > 0);
    //if(density<1e-4) density = 0.01;
    if(density<1e-4) {
        std::cout<<"density"<<density<<std::endl;
        std::abort();
    }
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
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
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
dealii::Tensor<2,nstate,real> Euler<dim,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // See Blazek Appendix A.9 p. 429-430
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel_normal = 0.0;
    for (int d=0;d<dim;d++) { vel_normal += vel[d] * normal[d]; }

    const real vel2 = compute_velocity_squared(vel);
    const real phi = 0.5*(gam-1.0) * vel2;

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
std::array<real,nstate> Euler<dim,nstate,real>
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
real Euler<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real sound = compute_sound (conservative_soln);
    const real vel2 = compute_velocity_squared(vel);
    const real max_eig = sqrt(vel2) + sound;
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

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, double> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    std::array<real,nstate> conservative_boundary_values;
    std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
    for (int s=0; s<nstate; s++) {
        conservative_boundary_values[s] = this->manufactured_solution_function.value (pos, s);
        boundary_gradients[s] = this->manufactured_solution_function.gradient (pos, s);
    }
    std::array<real,nstate> primitive_boundary_values = convert_conservative_to_primitive(conservative_boundary_values);

    if (boundary_type == 0) {
        for (int istate=0; istate<nstate; ++istate) {

            std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(conservative_boundary_values, normal_int);
            const bool inflow = (characteristic_dot_n[istate] <= 0.);

            if (inflow) { // Dirichlet boundary condition

                soln_bc[istate] = conservative_boundary_values[istate];
                soln_grad_bc[istate] = soln_grad_int[istate];

                // Only set the pressure and velocity
                // primitive_boundary_values[0] = soln_int[0];;
                // for(int d=0;d<dim;d++){
                //    primitive_boundary_values[1+d] = soln_int[1+d]/soln_int[0];;
                //}
                conservative_boundary_values = convert_primitive_to_conservative(primitive_boundary_values);
                //conservative_boundary_values[nstate-1] = soln_int[nstate-1];
                soln_bc[istate] = conservative_boundary_values[istate];

            } else { // Neumann boundary condition
                // soln_bc[istate] = -soln_int[istate]+2*conservative_boundary_values[istate];
                soln_bc[istate] = soln_int[istate];

                // **************************************************************************************************************
                // Note I don't know how to properly impose the soln_grad_bc to obtain an adjoint consistent scheme
                // Currently, Neumann boundary conditions are only imposed for the linear advection
                // Therefore, soln_grad_bc does not affect the solution
                // **************************************************************************************************************
                soln_grad_bc[istate] = soln_grad_int[istate];
                //soln_grad_bc[istate] = boundary_gradients[istate];
                //soln_grad_bc[istate] = -soln_grad_int[istate]+2*boundary_gradients[istate];
            }

            // HARDCODE DIRICHLET BC
            soln_bc[istate] = conservative_boundary_values[istate];

        }
    } else if (boundary_type == 1) {
        // No penetration,
        // Given by Algorithm II of the following paper
        // Krivodonova, L., and Berger, M.,
        // “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
        // Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
        const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(conservative_boundary_values);
        
        // Copy density and pressure
        std::array<real,nstate> primitive_boundary_values;
        primitive_boundary_values[0] = primitive_interior_values[0];
        primitive_boundary_values[nstate-1] = primitive_interior_values[nstate-1];

        const dealii::Tensor<1,dim,real> surface_normal = -normal_int;
        const dealii::Tensor<1,dim,real> velocities_int = extract_velocities_from_primitive(primitive_interior_values);
        const dealii::Tensor<1,dim,real> velocities_bc = velocities_int - 2.0*(velocities_int*surface_normal)*surface_normal;
        for (int d=0; d<dim; ++d) {
            primitive_boundary_values[1+d] = velocities_bc[d];
        }

        soln_bc = convert_primitive_to_conservative(primitive_boundary_values);

    } else if (boundary_type == 2) {
        // Pressure Outflow Boundary Condition (back pressure)
        // Carlson 2011, sec. 2.4

        const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(conservative_boundary_values);
        const real temperature_int = compute_temperature(primitive_interior_values);

    } else if (boundary_type == 3) {
    } else if (boundary_type == 4) {
    }
}

// Instantiate explicitly

template class Euler < PHILIP_DIM, PHILIP_DIM+2, double >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, Sacado::Fad::DFad<double>  >;

} // Physics namespace
} // PHiLiP namespace


