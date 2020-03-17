#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include "numerical_flux.h"
#include "viscous_numerical_flux.h"
#include "split_form_numerical_flux.h"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

// Protyping low level functions
template<int nstate, typename real_tensor>
std::array<real_tensor, nstate> array_average(
    const std::array<real_tensor, nstate> &array1,
    const std::array<real_tensor, nstate> &array2)
{
    std::array<real_tensor,nstate> array_average;
    for (int s=0; s<nstate; s++) {
        array_average[s] = 0.5*(array1[s] + array2[s]);
    }
    return array_average;
}


template <int dim, int nstate, typename real>
NumericalFluxConvective<dim,nstate,real>*
NumericalFluxFactory<dim, nstate, real>
::create_convective_numerical_flux(
    AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    if(conv_num_flux_type == AllParam::lax_friedrichs) {
        return new LaxFriedrichs<dim, nstate, real>(physics_input);
    } else if(conv_num_flux_type == AllParam::roe) {
        if constexpr (dim+2==nstate) return new Roe<dim, nstate, real>(physics_input);
    }
    else if (conv_num_flux_type == AllParam::split_form) {
        return new SplitFormNumFlux<dim, nstate, real>(physics_input);
    }

    std::cout << "Invalid numerical flux" << std::endl;
    return nullptr;
}
template <int dim, int nstate, typename real>
NumericalFluxDissipative<dim,nstate,real>*
NumericalFluxFactory<dim, nstate, real>
::create_dissipative_numerical_flux(
    AllParam::DissipativeNumericalFlux diss_num_flux_type,
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    if(diss_num_flux_type == AllParam::symm_internal_penalty) {
        return new SymmetricInternalPenalty<dim, nstate, real>(physics_input);
    }
    if(diss_num_flux_type == AllParam::BR2) {
        return new BassiRebay2<dim, nstate, real>(physics_input);
    }

    return nullptr;
}

template <int dim, int nstate, typename real>
NumericalFluxConvective<dim,nstate,real>::~NumericalFluxConvective() {}

template<int dim, int nstate, typename real>
std::array<real, nstate> LaxFriedrichs<dim,nstate,real>
::evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = pde_physics->convective_flux (soln_int);
    conv_phys_flux_ext = pde_physics->convective_flux (soln_ext);
    
    RealArrayVector flux_avg = array_average<nstate, dealii::Tensor<1,dim,real>> (conv_phys_flux_int, conv_phys_flux_ext);

//#if 0
    const real conv_max_eig_int = pde_physics->max_convective_eigenvalue(soln_int);
    const real conv_max_eig_ext = pde_physics->max_convective_eigenvalue(soln_ext);
    const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
//#endif
    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = flux_avg[s]*normal_int - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
        //numerical_flux_dot_n[s] = flux_avg[s]*normal_int; 
    }

    return numerical_flux_dot_n;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> Roe<dim,nstate,real>
::evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    // Blazek 2015
    // p. 103-105
    const std::array<real,nstate> prim_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> prim_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    // Left cell
    const real density_L = prim_soln_int[0];
    const dealii::Tensor< 1,dim,real > velocities_L = euler_physics->extract_velocities_from_primitive(prim_soln_int);
    const real pressure_L = prim_soln_int[nstate-1];

    const real normal_vel_L = velocities_L*normal_int;
    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, pressure_L);

    // Right cell
    const real density_R = prim_soln_ext[0];
    const dealii::Tensor< 1,dim,real > velocities_R = euler_physics->extract_velocities_from_primitive(prim_soln_ext);
    const real pressure_R = prim_soln_ext[nstate-1];

    const real normal_vel_R = velocities_R*normal_int;
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, pressure_R);

    // Roe-averaged states
    const real r = std::sqrt(density_R/density_L);
    const real rp1 = r+1.0;

    const real density_ravg = r*density_L;
    const dealii::Tensor< 1,dim,real > velocities_ravg = (r*velocities_R + velocities_L) / rp1;
    const real specific_total_enthalpy_ravg = (r*specific_enthalpy_R + specific_enthalpy_L) / rp1;

    const real vel2_ravg = euler_physics->compute_velocity_squared (velocities_ravg);
    const real normal_vel_ravg = velocities_ravg*normal_int;

    const real sound2_ravg = euler_physics->gamm1*(specific_total_enthalpy_ravg-0.5*vel2_ravg);
    const real sound_ravg = std::sqrt(sound2_ravg);

    // Compute eigenvalues
    std::array<real, 3> eig_ravg;
    eig_ravg[0] = std::abs(normal_vel_ravg-sound_ravg);
    eig_ravg[1] = std::abs(normal_vel_ravg);
    eig_ravg[2] = std::abs(normal_vel_ravg+sound_ravg);

    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    std::array<real, 3> eig_L;
    eig_L[0] = std::abs(normal_vel_L-sound_L);
    eig_L[1] = std::abs(normal_vel_L);
    eig_L[2] = std::abs(normal_vel_L+sound_L);

    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
    std::array<real, 3> eig_R;
    eig_R[0] = std::abs(normal_vel_R-sound_R);
    eig_R[1] = std::abs(normal_vel_R);
    eig_R[2] = std::abs(normal_vel_R+sound_R);

    // Harten's entropy fix
    for(int e=0;e<3;e++) {
        const real eps = std::max(std::abs(eig_ravg[e]-eig_L[e]), std::abs(eig_R[e]-eig_ravg[e]));
        if(std::abs(eig_ravg[e]) < eps) {
            eig_ravg[e] = 0.5*(eig_ravg[e]*eig_ravg[e]/eps + eps);
        }
    }

    // Physical fluxes
    const std::array<real,nstate> normal_flux_int = euler_physics->convective_normal_flux (soln_int, normal_int);
    const std::array<real,nstate> normal_flux_ext = euler_physics->convective_normal_flux (soln_ext, normal_int);

    const real dVn = normal_vel_R-normal_vel_L;
    const real dp = pressure_R - pressure_L;
    const real drho = density_R - density_L;

    // Product of eigenvalues and wave strengths
    real coeff[4];
    coeff[0] = eig_ravg[0]*(dp-density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);
    coeff[1] = eig_ravg[1]*(drho - dp/sound2_ravg);
    coeff[2] = eig_ravg[1]*density_ravg;
    coeff[3] = eig_ravg[2]*(dp+density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);

    // Evaluate |A_Roe| * (W_R - W_L)
    std::array<real,nstate> AdW;

    // Vn-c
    AdW[0] = coeff[0] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] = coeff[0] * (velocities_ravg[d] - sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] = coeff[0] * (specific_total_enthalpy_ravg - sound_ravg*normal_vel_ravg);

    // Vn
    AdW[0] += coeff[1] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[1] * velocities_ravg[d];
    }
    AdW[nstate-1] += coeff[1] * vel2_ravg * 0.5;

    AdW[0] += coeff[2] * 0.0;
    const dealii::Tensor<1,dim,real> dvel = velocities_R - velocities_L;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[2] * (dvel[d] - dVn*normal_int[d]);
    }
    AdW[nstate-1] += coeff[2] * (velocities_ravg*dvel - normal_vel_ravg*dVn);

    // Vn+c
    AdW[0] += coeff[3] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[3] * (velocities_ravg[d] + sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] += coeff[3] * (specific_total_enthalpy_ravg + sound_ravg*normal_vel_ravg);

    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = 0.5*(normal_flux_int[s]+normal_flux_ext[s] - AdW[s]);
    }

    return numerical_flux_dot_n;
}


// Instantiation
template class NumericalFluxConvective<PHILIP_DIM, 1, double>;
template class NumericalFluxConvective<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 2, double>;
template class NumericalFluxConvective<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 3, double>;
template class NumericalFluxConvective<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 4, double>;
template class NumericalFluxConvective<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 5, double>;
template class NumericalFluxConvective<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

template class LaxFriedrichs<PHILIP_DIM, 1, double>;
template class LaxFriedrichs<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 2, double>;
template class LaxFriedrichs<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 3, double>;
template class LaxFriedrichs<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 4, double>;
template class LaxFriedrichs<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 5, double>;
template class LaxFriedrichs<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

template class Roe<PHILIP_DIM, PHILIP_DIM+2, double>;
template class Roe<PHILIP_DIM, PHILIP_DIM+2, Sacado::Fad::DFad<double> >;


template class NumericalFluxFactory<PHILIP_DIM, 1, double>;
template class NumericalFluxFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 2, double>;
template class NumericalFluxFactory<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 3, double>;
template class NumericalFluxFactory<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 4, double>;
template class NumericalFluxFactory<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 5, double>;
template class NumericalFluxFactory<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;


} // NumericalFlux namespace
} // PHiLiP namespace
