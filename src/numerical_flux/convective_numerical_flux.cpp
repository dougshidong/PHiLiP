#include "ADTypes.hpp"

#include "convective_numerical_flux.hpp"

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
    
    //RealArrayVector flux_avg = array_average<nstate, dealii::Tensor<1,dim,real>> (conv_phys_flux_int, conv_phys_flux_ext);
    RealArrayVector flux_avg;
    for (int s=0; s<nstate; s++) {
        flux_avg[s] = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_avg[s][d] = 0.5*(conv_phys_flux_int[s][d] + conv_phys_flux_ext[s][d]);
        }
    }

    const real conv_max_eig_int = pde_physics->max_convective_eigenvalue(soln_int);
    const real conv_max_eig_ext = pde_physics->max_convective_eigenvalue(soln_ext);
    // Replaced the std::max with an if-statement for the AD to work properly.
    //const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
    real conv_max_eig;
    if (conv_max_eig_int > conv_max_eig_ext) {
        conv_max_eig = conv_max_eig_int;
    } else {
        conv_max_eig = conv_max_eig_ext;
    }
    //conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        //numerical_flux_dot_n[s] = flux_avg[s]*normal_int - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
        real flux_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_dot_n += flux_avg[s][d]*normal_int[d];
        }
        numerical_flux_dot_n[s] = flux_dot_n - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
    }

    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
void RoePike<dim,nstate,real>
::evaluate_entropy_fix (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    std::array<real, 3> &eig_RoeAvg,
    const real /*vel2_ravg*/,
    const real /*sound_ravg*/) const
{
    // Harten's entropy fix
    // -- See Blazek 2015, p.103-105
    for(int e=0;e<3;e++) {
        const real eps = std::max(abs(eig_RoeAvg[e] - eig_L[e]), abs(eig_R[e] - eig_RoeAvg[e]));
        if(eig_RoeAvg[e] < eps) {
            eig_RoeAvg[e] = 0.5*(eig_RoeAvg[e] * eig_RoeAvg[e]/eps + eps);
        }
    }
}

template <int dim, int nstate, typename real>
void RoePike<dim,nstate,real>
::evaluate_additional_modifications (
    const std::array<real, nstate> &/*soln_int*/,
    const std::array<real, nstate> &/*soln_ext*/,
    const std::array<real, 3> &/*eig_L*/,
    const std::array<real, 3> &/*eig_R*/,
    real &/*dV_normal*/, 
    dealii::Tensor<1,dim,real> &/*dV_tangent*/
    ) const
{
    // No additional modifications for the Roe-Pike scheme
}

template <int dim, int nstate, typename real>
void L2Roe<dim,nstate,real>
::evaluate_shock_indicator (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    int &ssw_LEFT,
    int &ssw_RIGHT) const
{
    // Shock indicator of Wada & Liou (1994 Flux) -- Eq.(39)
    // -- See also p.74 of Osswald et al. (2016 L2Roe)
    
    ssw_LEFT=0; ssw_RIGHT=0; // initialize
    
    // ssw_L: i=L --> j=R
    if((eig_L[0]>0.0 && eig_R[0]<0.0) || (eig_L[2]>0.0 && eig_R[2]<0.0)) {
        ssw_LEFT = 1;
    }
    
    // ssw_R: i=R --> j=L
    if((eig_R[0]>0.0 && eig_L[0]<0.0) || (eig_R[2]>0.0 && eig_L[2]<0.0)) {
        ssw_RIGHT = 1;
    }
}

template <int dim, int nstate, typename real>
void L2Roe<dim,nstate,real>
::evaluate_entropy_fix (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    std::array<real, 3> &eig_RoeAvg,
    const real vel2_ravg,
    const real sound_ravg) const
{
    // Van Leer et al. (1989 Sonic) entropy fix for acoustic waves
    // -- p.74 of Osswald et al. (2016 L2Roe)
    for(int e=0;e<3;e++) {
        if(e!=1) {
            // const real deig = std::max((eig_R[e]-eig_L[e]), 0.0);
            const real deig = std::max(static_cast<real>(eig_R[e] - eig_L[e]), static_cast<real>(0.0));
            if(eig_RoeAvg[e] < 2.0*deig) {
                eig_RoeAvg[e] = 0.25*(eig_RoeAvg[e] * eig_RoeAvg[e]/deig) + deig;
            }
        }
    }
    
    // Entropy fix of Liou (2000 Mass)
    // -- p.74 of Osswald et al. (2016 L2Roe)
    int ssw_L, ssw_R;
    evaluate_shock_indicator(eig_L,eig_R,ssw_L,ssw_R);
    if(ssw_L!=0 || ssw_R!=0) {
        eig_RoeAvg[1] = std::max(sound_ravg, static_cast<real>(sqrt(vel2_ravg)));
    }
}

template <int dim, int nstate, typename real>
void L2Roe<dim,nstate,real>
::evaluate_additional_modifications  (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    real &dV_normal, 
    dealii::Tensor<1,dim,real> &dV_tangent) const
{
    const real mach_number_L = this->euler_physics->compute_mach_number(soln_int);
    const real mach_number_R = this->euler_physics->compute_mach_number(soln_ext);

    // Osswald's two modifications to Roe-Pike scheme --> L2Roe
    // - Blending factor (variable 'z' in reference)
    const real blending_factor = std::min(static_cast<real>(1.0), std::max(mach_number_L,mach_number_R));
    // - Scale jump in (1) normal and (2) tangential velocities
    int ssw_L, ssw_R;
    evaluate_shock_indicator(eig_L,eig_R,ssw_L,ssw_R);
    if(ssw_L==0 && ssw_R==0)
    {
        dV_normal *= blending_factor;
        for (int d=0;d<dim;d++)
        {
            dV_tangent[d] *= blending_factor;
        }
    }
}

template <int dim, int nstate, typename real>
std::array<real, nstate> RoeBase<dim,nstate,real>
::evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    // See Blazek 2015, p.103-105
    // -- Note: Modified calculation of alpha_{3,4} to use 
    //          dVt (jump in tangential velocities);
    //          expressions are equivalent
    
    // Blazek 2015
    // p. 103-105
    // Note: This is in fact the Roe-Pike method of Roe & Pike (1984 - Efficient)
    const std::array<real,nstate> prim_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> prim_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    // Left cell
    const real density_L = prim_soln_int[0];
    const dealii::Tensor< 1,dim,real > velocities_L = euler_physics->extract_velocities_from_primitive(prim_soln_int);
    const real pressure_L = prim_soln_int[nstate-1];

    //const real normal_vel_L = velocities_L*normal_int;
    real normal_vel_L = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_L+= velocities_L[d]*normal_int[d];
    }
    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, pressure_L);

    // Right cell
    const real density_R = prim_soln_ext[0];
    const dealii::Tensor< 1,dim,real > velocities_R = euler_physics->extract_velocities_from_primitive(prim_soln_ext);
    const real pressure_R = prim_soln_ext[nstate-1];

    //const real normal_vel_R = velocities_R*normal_int;
    real normal_vel_R = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_R+= velocities_R[d]*normal_int[d];
    }
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, pressure_R);

    // Roe-averaged states
    const real r = sqrt(density_R/density_L);
    const real rp1 = r+1.0;

    const real density_ravg = r*density_L;
    //const dealii::Tensor< 1,dim,real > velocities_ravg = (r*velocities_R + velocities_L) / rp1;
    dealii::Tensor< 1,dim,real > velocities_ravg;
    for (int d=0; d<dim; ++d) {
        velocities_ravg[d] = (r*velocities_R[d] + velocities_L[d]) / rp1;
    }
    const real specific_total_enthalpy_ravg = (r*specific_enthalpy_R + specific_enthalpy_L) / rp1;

    const real vel2_ravg = euler_physics->compute_velocity_squared (velocities_ravg);
    //const real normal_vel_ravg = velocities_ravg*normal_int;
    real normal_vel_ravg = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_ravg += velocities_ravg[d]*normal_int[d];
    }

    const real sound2_ravg = euler_physics->gamm1*(specific_total_enthalpy_ravg-0.5*vel2_ravg);
    real sound_ravg = 1e10;
    if (sound2_ravg > 0.0) {
        sound_ravg = sqrt(sound2_ravg);
    }

    // Compute eigenvalues
    std::array<real, 3> eig_ravg;
    eig_ravg[0] = abs(normal_vel_ravg-sound_ravg);
    eig_ravg[1] = abs(normal_vel_ravg);
    eig_ravg[2] = abs(normal_vel_ravg+sound_ravg);

    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    std::array<real, 3> eig_L;
    eig_L[0] = abs(normal_vel_L-sound_L);
    eig_L[1] = abs(normal_vel_L);
    eig_L[2] = abs(normal_vel_L+sound_L);

    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
    std::array<real, 3> eig_R;
    eig_R[0] = abs(normal_vel_R-sound_R);
    eig_R[1] = abs(normal_vel_R);
    eig_R[2] = abs(normal_vel_R+sound_R);

    // Jumps in pressure and density
    const real dp = pressure_R - pressure_L;
    const real drho = density_R - density_L;

    // Jump in normal velocity
    real dVn = normal_vel_R-normal_vel_L;

    // Jumps in tangential velocities
    dealii::Tensor<1,dim,real> dVt;
    for (int d=0;d<dim;d++) {
        dVt[d] = (velocities_R[d] - velocities_L[d]) - dVn*normal_int[d];
    }

    // Evaluate entropy fix on wave speeds
    evaluate_entropy_fix (eig_L, eig_R, eig_ravg, vel2_ravg, sound_ravg);

    // Evaluate additional modifications to the Roe-Pike scheme (if applicable)
    evaluate_additional_modifications (soln_int, soln_ext, eig_L, eig_R, dVn, dVt);

    // Physical fluxes
    const std::array<real,nstate> normal_flux_int = euler_physics->convective_normal_flux (soln_int, normal_int);
    const std::array<real,nstate> normal_flux_ext = euler_physics->convective_normal_flux (soln_ext, normal_int);

    // Product of eigenvalues and wave strengths
    real coeff[4];
    coeff[0] = eig_ravg[0]*(dp-density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);
    coeff[1] = eig_ravg[1]*(drho - dp/sound2_ravg);
    coeff[2] = eig_ravg[1]*density_ravg;
    coeff[3] = eig_ravg[2]*(dp+density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);

    // Evaluate |A_Roe| * (W_R - W_L)
    std::array<real,nstate> AdW;

    // Vn-c (i=1)
    AdW[0] = coeff[0] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] = coeff[0] * (velocities_ravg[d] - sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] = coeff[0] * (specific_total_enthalpy_ravg - sound_ravg*normal_vel_ravg);

    // Vn (i=2)
    AdW[0] += coeff[1] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[1] * velocities_ravg[d];
    }
    AdW[nstate-1] += coeff[1] * vel2_ravg * 0.5;

    // (i=3,4)
    AdW[0] += coeff[2] * 0.0;
    real dVt_dot_vel_ravg = 0.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[2]*dVt[d];
        dVt_dot_vel_ravg += velocities_ravg[d]*dVt[d];
    }
    AdW[nstate-1] += coeff[2]*dVt_dot_vel_ravg;

    // Vn+c (i=5)
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
template class NumericalFluxConvective<PHILIP_DIM, 2, double>;
template class NumericalFluxConvective<PHILIP_DIM, 3, double>;
template class NumericalFluxConvective<PHILIP_DIM, 4, double>;
template class NumericalFluxConvective<PHILIP_DIM, 5, double>;
template class NumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, RadType >;

template class NumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;

template class LaxFriedrichs<PHILIP_DIM, 1, double>;
template class LaxFriedrichs<PHILIP_DIM, 2, double>;
template class LaxFriedrichs<PHILIP_DIM, 3, double>;
template class LaxFriedrichs<PHILIP_DIM, 4, double>;
template class LaxFriedrichs<PHILIP_DIM, 5, double>;
template class LaxFriedrichs<PHILIP_DIM, 1, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, RadFadType >;

template class RoeBase<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoeBase<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoeBase<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoeBase<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoeBase<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class RoePike<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, double>;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // NumericalFlux namespace
} // PHiLiP namespace
