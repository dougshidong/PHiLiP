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
NumericalFluxConvective<dim, nstate, real>::NumericalFluxConvective(
    std::unique_ptr< BaselineNumericalFluxConvective<dim,nstate,real> > baseline_input,
    std::unique_ptr< RiemannSolverDissipation<dim,nstate,real> >   riemann_solver_dissipation_input)
    : baseline(std::move(baseline_input))
    , riemann_solver_dissipation(std::move(riemann_solver_dissipation_input))
{ }

template<int dim, int nstate, typename real>
std::array<real, nstate> NumericalFluxConvective<dim,nstate,real>
::evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    // baseline flux (without upwind dissipation)
    const std::array<real, nstate> baseline_flux_dot_n 
        = this->baseline->evaluate_flux(soln_int, soln_ext, normal_int);

    // Riemann solver dissipation
    const std::array<real, nstate> riemann_solver_dissipation_dot_n 
        = this->riemann_solver_dissipation->evaluate_riemann_solver_dissipation(soln_int, soln_ext, normal_int);

    // convective numerical flux: sum of baseline and Riemann solver dissipation term
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = baseline_flux_dot_n[s] + riemann_solver_dissipation_dot_n[s];
    }
    return numerical_flux_dot_n;
}

template<int dim, int nstate, typename real>
void NumericalFluxConvective<dim,nstate,real>::set_upwinding_flux(const bool _use_upwinding)
{
    this->baseline->set_upwinding_flux(_use_upwinding);
}

template <int dim, int nstate, typename real>
LaxFriedrichs<dim, nstate, real>::LaxFriedrichs(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< LaxFriedrichsRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
RoePike<dim, nstate, real>::RoePike(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< RoePikeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
L2Roe<dim, nstate, real>::L2Roe(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< L2RoeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
HLLC<dim, nstate, real>::HLLC(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< HLLCBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< ZeroRiemannSolverDissipation<dim, nstate, real> > ())
{}

template <int dim, int nstate, typename real>
Central<dim, nstate, real>::Central(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< ZeroRiemannSolverDissipation<dim, nstate, real> > ())
{}

template <int dim, int nstate, typename real>
EntropyConserving<dim, nstate, real>::EntropyConserving(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< ZeroRiemannSolverDissipation<dim, nstate, real> > ())
{}

template <int dim, int nstate, typename real>
EntropyConservingWithLaxFriedrichsDissipation<dim, nstate, real>::EntropyConservingWithLaxFriedrichsDissipation(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< LaxFriedrichsRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
EntropyConservingWithRoeDissipation<dim, nstate, real>::EntropyConservingWithRoeDissipation(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< RoePikeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
EntropyConservingWithL2RoeDissipation<dim, nstate, real>::EntropyConservingWithL2RoeDissipation(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< L2RoeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
BaselineNumericalFluxConvective<dim,nstate,real>::~BaselineNumericalFluxConvective() {}

template <int dim, int nstate, typename real>
std::array<real, nstate> CentralBaselineNumericalFluxConvective<dim,nstate,real>::evaluate_flux(
 const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = pde_physics->convective_flux (soln_int);
    conv_phys_flux_ext = pde_physics->convective_flux (soln_ext);
    
    RealArrayVector flux_avg;
    for (int s=0; s<nstate; s++) {
        flux_avg[s] = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_avg[s][d] = 0.5*(conv_phys_flux_int[s][d] + conv_phys_flux_ext[s][d]);
        }
    }

    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        real flux_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_dot_n += flux_avg[s][d]*normal_int[d];
        }
        numerical_flux_dot_n[s] = flux_dot_n;
    }
    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
std::array<real, nstate> EntropyConservingBaselineNumericalFluxConvective<dim,nstate,real>::evaluate_flux(
 const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_split_flux;

    conv_phys_split_flux = pde_physics->convective_numerical_split_flux (soln_int,soln_ext);

    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        real flux_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_dot_n += conv_phys_split_flux[s][d] * normal_int[d];
        }
        numerical_flux_dot_n[s] = flux_dot_n;
    }
    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
void HLLCBaselineNumericalFluxConvective<dim,nstate,real>::set_upwinding_flux(const bool _use_upwinding)
{
    use_upwinding = _use_upwinding;
}

template <int dim, int nstate, typename real>
std::array<real, nstate> HLLCBaselineNumericalFluxConvective<dim,nstate,real>::evaluate_flux(
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    // Using HLLC from Appendix B of Yu Lv and Matthias Ihme, 2014, Discontinuous Galerkin method for 
    // multicomponent chemically reacting ﬂows and combustion.
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    
    const std::array<real,nstate> prim_soln_L = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> prim_soln_R = euler_physics->convert_conservative_to_primitive(soln_ext);

    const real density_L = prim_soln_L[0];
    const real density_R = prim_soln_R[0];
    const real pressure_L = prim_soln_L[nstate-1];
    const real pressure_R = prim_soln_R[nstate-1];
    const dealii::Tensor< 1,dim,real > velocities_L = euler_physics->extract_velocities_from_primitive(prim_soln_L);
    const dealii::Tensor< 1,dim,real > velocities_R = euler_physics->extract_velocities_from_primitive(prim_soln_R);
    real velocity_dot_n_L = 0;
    real velocity_dot_n_R = 0;
    for(int d=0; d<dim; ++d)
    {
        velocity_dot_n_L += velocities_L[d]*normal_int[d];
        velocity_dot_n_R += velocities_R[d]*normal_int[d];
    }
    
    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
/*
    // Pressure PVRS approach
    const real density_avg = 0.5*(density_L + density_R);
    const real sound_avg = 0.5*(sound_L + sound_R);
    const real pressure_pvrs = 0.5*(pressure_L + pressure_R) - 0.5*(velocity_dot_n_R - velocity_dot_n_L)*density_avg*sound_avg;
    real pressure_star = 0;
    if(pressure_pvrs > 0.0)
    {
        pressure_star = pressure_pvrs;
    }
*/

    // Hybrid scheme
    const real density_avg = 0.5*(density_L + density_R);
    const real sound_avg = 0.5*(sound_L + sound_R);
    const real pressure_pvrs = 0.5*(pressure_L + pressure_R) - 0.5*(velocity_dot_n_R - velocity_dot_n_L)*density_avg*sound_avg;
    real p_min = pressure_L;
    if(pressure_R<p_min)
    {
        p_min = pressure_R;
    }
    real pressure_star = 0;
    if(pressure_pvrs <= p_min)
    {
        // Two–Rarefaction Riemann solver TRRS approach
        const real zval = (euler_physics->gam - 1.0)/(2.0*euler_physics->gam);
        const real kernelval = (sound_L + sound_R - (euler_physics->gam-1.0)/2.0*(velocity_dot_n_R - velocity_dot_n_L)) 
                                / (sound_L/pow(pressure_L,zval) + sound_R/pow(pressure_R,zval));
        pressure_star = pow(kernelval,1.0/zval);
    }
    else
    {
        // Two–Shock Riemann solver TSRS approach
        real p0 = 0.0;
        if(pressure_pvrs>0.0)
        {
            p0 = pressure_pvrs;
        }
        const real A_L = 2.0/((euler_physics->gam+1)*density_L);
        const real A_R = 2.0/((euler_physics->gam+1)*density_R);
        const real B_L = (euler_physics->gam - 1.0)/(euler_physics->gam + 1.0)*pressure_L;
        const real B_R = (euler_physics->gam - 1.0)/(euler_physics->gam + 1.0)*pressure_R;
        const real g_L = sqrt(A_L/(p0 + B_L));
        const real g_R = sqrt(A_R/(p0 + B_R));
        pressure_star = (g_L*pressure_L + g_R*pressure_R - (velocity_dot_n_R - velocity_dot_n_L))/(g_L+g_R);
    }
    // Kernel of pressure-based approach.
    const real gam_fraction = (euler_physics->gam + 1.0)/(2.0*euler_physics->gam);
    real q_L = 1.0;
    if(pressure_star > pressure_L)
    {
        const real val = 1.0 + gam_fraction*(pressure_star/pressure_L - 1.0);
        q_L = sqrt(val);
    }
    real q_R = 1.0;
    if(pressure_star > pressure_R)
    {
        const real val = 1.0 + gam_fraction*(pressure_star/pressure_R - 1.0);
        q_R = sqrt(val);
    }

    real S_L = velocity_dot_n_L - sound_L*q_L;
    real S_R = velocity_dot_n_R + sound_R*q_R;

/*
    // Einfieldt's approach
    const real eta2 = 0.5 * sqrt(density_L)*sqrt(density_R)/(pow(sqrt(density_L) + sqrt(density_R),2));
    const real dbar_squared = (sqrt(density_L)*pow(sound_L,2) + sqrt(density_R)*pow(sound_R,2))/(sqrt(density_L) + sqrt(density_R))
                                + eta2*pow(velocity_dot_n_R - velocity_dot_n_L,2);
    const real dbar = sqrt(dbar_squared);
    const real ubar = (sqrt(density_L)*velocity_dot_n_L + sqrt(density_R)*velocity_dot_n_R)/(sqrt(density_L) + sqrt(density_R));
    real S_L = ubar - dbar;
    real S_R = ubar + dbar;
*/
    if(use_upwinding)
    {
        // Simple Davis approach
        S_L = velocity_dot_n_L - sound_L;
        S_R = velocity_dot_n_R + sound_R;
    }

/*
    // Using Roe based approaximations.
    (void) sound_L; (void) sound_R;
    const real ubar = (sqrt(density_L)*velocity_dot_n_L + sqrt(density_R)*velocity_dot_n_R)/(sqrt(density_L) + sqrt(density_R));
    const real enthalpy_bar = (sqrt(density_L)*euler_physics->compute_specific_enthalpy(soln_int, density_L) 
                            + sqrt(density_R)*euler_physics->compute_specific_enthalpy(soln_ext, density_R))/(sqrt(density_L) + sqrt(density_R));
    const real sound_bar = sqrt(euler_physics->gamm1*(enthalpy_bar - 0.5*pow(ubar,2)));
    const real S_L = ubar - sound_bar;
    const real S_R = ubar + sound_bar;
*/

    const real S_star = 
            (pressure_R - pressure_L + density_L*velocity_dot_n_L*(S_L - velocity_dot_n_L) 
            - density_R*velocity_dot_n_R*(S_R - velocity_dot_n_R))/(density_L*(S_L - velocity_dot_n_L) - density_R*(S_R - velocity_dot_n_R));
    

    std::array<real, nstate> soln_star_L;
    std::array<real, nstate> soln_star_R;
    const real multfactor_L = (S_L - velocity_dot_n_L)/(S_L - S_star);
    const real multfactor_R = (S_R - velocity_dot_n_R)/(S_R - S_star);

    soln_star_L[0] = soln_int[0];
    soln_star_R[0] = soln_ext[0];

    for(int d=0; d<dim; ++d)
    {
        soln_star_L[1+d] = soln_int[1+d] + density_L*(S_star - velocity_dot_n_L)*normal_int[d];
        
        soln_star_R[1+d] = soln_ext[1+d] + density_R*(S_star - velocity_dot_n_R)*normal_int[d];
    }

    soln_star_L[nstate-1] = soln_int[nstate-1] + (S_star - velocity_dot_n_L)*(density_L*S_star + pressure_L/(S_L - velocity_dot_n_L));
    soln_star_R[nstate-1] = soln_ext[nstate-1] + (S_star - velocity_dot_n_R)*(density_R*S_star + pressure_R/(S_R - velocity_dot_n_R));

    for(int s=0; s<nstate; ++s)
    {
        soln_star_L[s] *= multfactor_L;
        soln_star_R[s] *= multfactor_R;
    }

    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = euler_physics->convective_flux (soln_int);
    conv_phys_flux_ext = euler_physics->convective_flux (soln_ext);
    

    std::array<real, nstate> numerical_flux_dot_n_L;
    std::array<real, nstate> numerical_flux_dot_n_R;

    for(int s = 0; s<nstate; ++s)
    {
        real flux_dot_n_L = 0.0;
        real flux_dot_n_R = 0.0;
        for(int d=0; d<dim; ++d)
        {
            flux_dot_n_L += conv_phys_flux_int[s][d]*normal_int[d];
            flux_dot_n_R += conv_phys_flux_ext[s][d]*normal_int[d];
        }
        numerical_flux_dot_n_L[s] = flux_dot_n_L;
        numerical_flux_dot_n_R[s] = flux_dot_n_R;
    }


    std::array<real, nstate> numerical_flux_dot_n;

    if( (S_L >= 0.0) )
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_L[s];
        }
    }
    else if( (S_R <= 0.0) )
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_R[s];
        } 
    }
/*
    // Shock-stable modiﬁcation of the HLLC Riemann solver.
    else
    {
        const real Ma_limit = 0.1;
        const real Ma_local = std::max(abs(velocity_dot_n_L/sound_L), abs(velocity_dot_n_R/sound_R));
        const real pi = 4.0*atan(1.0);
        real min_val = 1.0;
        if((Ma_local/Ma_limit) < min_val)
        {
            min_val = (Ma_local/Ma_limit);
        }
        const real phi = sin(min_val*pi/2.0);
        const real S_L_lm = phi*S_L;
        const real S_R_lm = phi*S_R;
        
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = 0.5*(numerical_flux_dot_n_L[s] + numerical_flux_dot_n_R[s]) +
                0.5*(S_L_lm*(soln_star_L[s] - soln_int[s]) + abs(S_star)*(soln_star_L[s] - soln_star_R[s]) + S_R_lm*(soln_star_R[s] - soln_ext[s]));

        }
    }
*/


    else if( (S_L <= 0.0) && (0.0 < S_star))
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_L[s] + S_L*(soln_star_L[s] - soln_int[s]);
        } 
    }
    else if( (S_star <= 0.0) && (0.0 < S_R))
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_R[s] + S_R*(soln_star_R[s] - soln_ext[s]);
        } 
    }
    else
    {
        std::cout<<"Shouldn't have reached here in HLLC flux."<<std::endl;
        std::abort();
    }

    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
RiemannSolverDissipation<dim,nstate,real>::~RiemannSolverDissipation() {}

template<int dim, int nstate, typename real>
std::array<real, nstate> ZeroRiemannSolverDissipation<dim,nstate,real>
::evaluate_riemann_solver_dissipation (
    const std::array<real, nstate> &/*soln_int*/,
    const std::array<real, nstate> &/*soln_ext*/,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    // zero upwind dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    numerical_flux_dot_n.fill(0.0);
    return numerical_flux_dot_n;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> LaxFriedrichsRiemannSolverDissipation<dim,nstate,real>
::evaluate_riemann_solver_dissipation (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
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
        numerical_flux_dot_n[s] = - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
    }    

    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
void RoePikeRiemannSolverDissipation<dim,nstate,real>
::evaluate_entropy_fix (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    std::array<real, 3> & eig_RoeAvg,
    const real /*vel2_ravg*/,
    const real /*sound_ravg*/,
    const real sound_L,
    const real sound_R,
    const real pressure_L,
    const real pressure_R) const
{
    (void) sound_L; (void) sound_R; (void) pressure_L; (void) pressure_R;
    // Harten-Hyman
    for(int i=0; i<3; ++i)
    {
        real delta_k = 0.0;
        if( (eig_RoeAvg[i] - eig_L[i]) > delta_k)
        {
            delta_k = (eig_RoeAvg[i] - eig_L[i]);
        }
        if( (eig_R[i] - eig_RoeAvg[i]) > delta_k)
        {
            delta_k = (eig_R[i] - eig_RoeAvg[i]);
        }

        if(abs(eig_RoeAvg[i]) < delta_k)
        {
            eig_RoeAvg[i] = 0.5*(eig_RoeAvg[i]*eig_RoeAvg[i]/delta_k + delta_k);
        }
        else
        {
            eig_RoeAvg[i] = abs(eig_RoeAvg[i]);
        }
    }
/*
    Hartmann
    real max_eig = -1.0;
    for(int i=0; i<3; ++i)
    {
       if(max_eig < eig_RoeAvg[i])
       {
           max_eig = eig_RoeAvg[i]; 
       }
    }
    const real delta = 0.1*max_eig;
    for(int i=0; i<3; ++i)
    {
        if(eig_RoeAvg[i] < delta)
        {
            eig_RoeAvg[i] = (eig_RoeAvg[i]*eig_RoeAvg[i] + delta*delta)/(2.0*delta);
        }
    }
*/
/*
    Torro
    const real u_L = eig_L[2] - sound_L;
    const real u_R = eig_R[2] - sound_R;
    const double gamma_val = 1.4;
    const double z_val = (gamma_val - 1.0)/(2.0*gamma_val);

    const real numerator = sound_L + sound_R - (gamma_val-1.0)/2.0*(u_R - u_L);
    const real denominator = sound_L/pow(pressure_L,z_val) + sound_R/pow(pressure_R, z_val);
    const real p_star = pow(numerator/denominator, 1.0/z_val);
    
    {
        // Verify if left wave u-a is a sonic rarefaction and update eigenvalue if it is.
        const real a_star_L = sound_L*pow(p_star/pressure_L,z_val);
        const real u_star_L = u_L + 2.0/(gamma_val-1.0)*(sound_L - a_star_L);

        const real lambda1_L = u_L - sound_L;
        const real lambda1_star_L = u_star_L - a_star_L;
        if( (lambda1_L < 0.0) && (lambda1_star_L > 0.0) )
        {
            // sonic rarefaction
            const real lambda1_bar = eig_RoeAvg[0];
            const real lambda1_new_L = lambda1_L*(lambda1_star_L - lambda1_bar)/(lambda1_star_L - lambda1_L);
            const real lambda1_new_R = lambda1_star_L*(lambda1_bar - lambda1_L)/(lambda1_star_L - lambda1_L);
            eig_RoeAvg[0] = abs(lambda1_new_R) + abs(lambda1_new_L);
        }
    }

    {
        // Verify if right wave u+a is a sonic rarefaction and update eigenvalue if it is.
        const real a_star_R = sound_R*pow(p_star/pressure_R,z_val);
        const real u_star_R = u_R + 2.0/(gamma_val-1.0)*(a_star_R - sound_R);

        const real lambda5_star_R = u_star_R + a_star_R;
        const real lambda5_R = u_R + sound_R;

        if( (lambda5_star_R<0.0) && (lambda5_R > 0.0))
        {
            // sonic rarefaction at right wave
            const real lambda5_bar = eig_RoeAvg[2];
            const real lambda5_new_star = lambda5_star_R*(lambda5_R - lambda5_bar)/(lambda5_R - lambda5_star_R);
            const real lambda5_new_R = lambda5_R*(lambda5_bar - lambda5_star_R)/(lambda5_R - lambda5_star_R);
            eig_RoeAvg[2] = abs(lambda5_new_star) + abs(lambda5_new_R);
        }
    }
*/
}

template <int dim, int nstate, typename real>
void RoePikeRiemannSolverDissipation<dim,nstate,real>
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
void L2RoeRiemannSolverDissipation<dim,nstate,real>
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
void L2RoeRiemannSolverDissipation<dim,nstate,real>
::evaluate_entropy_fix (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    std::array<real, 3> &eig_RoeAvg,
    const real vel2_ravg,
    const real sound_ravg,
    const real /*sound_L*/,
    const real /*sound_R*/,
    const real /*pressure_L*/,
    const real /*pressure_R*/) const
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
void L2RoeRiemannSolverDissipation<dim,nstate,real>
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
std::array<real, nstate> RoeBaseRiemannSolverDissipation<dim,nstate,real>
::evaluate_riemann_solver_dissipation (
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
    eig_ravg[0] = normal_vel_ravg-sound_ravg;
    eig_ravg[1] = normal_vel_ravg;
    eig_ravg[2] = normal_vel_ravg+sound_ravg;

    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    std::array<real, 3> eig_L;
    eig_L[0] = normal_vel_L-sound_L;
    eig_L[1] = normal_vel_L;
    eig_L[2] = normal_vel_L+sound_L;

    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
    std::array<real, 3> eig_R;
    eig_R[0] = normal_vel_R-sound_R;
    eig_R[1] = normal_vel_R;
    eig_R[2] = normal_vel_R+sound_R;

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
    evaluate_entropy_fix (eig_L, eig_R, eig_ravg, vel2_ravg, sound_ravg, sound_L, sound_R, pressure_L, pressure_R);
    
    for(int i=0; i<3; ++i)
    {
        eig_ravg[i] = abs(eig_ravg[i]);
        eig_L[i] = abs(eig_L[i]);
        eig_R[i] = abs(eig_R[i]);
    }

    // Evaluate additional modifications to the Roe-Pike scheme (if applicable)
    evaluate_additional_modifications (soln_int, soln_ext, eig_L, eig_R, dVn, dVt);

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
        numerical_flux_dot_n[s] = - 0.5 * AdW[s];
    }

    return numerical_flux_dot_n;
/*
//=====================================================================================================================
//                                          CUSP SCHEME
//=====================================================================================================================

    //CUSP scheme
    const real pressure_int = euler_physics->compute_pressure(soln_int);
    const real pressure_ext = euler_physics->compute_pressure(soln_ext);
    const real specific_total_enthalpy_int = soln_int[nstate-1] / soln_int[0] + pressure_int / soln_int[0];
    const real specific_total_enthalpy_ext = soln_ext[nstate-1] / soln_ext[0] + pressure_ext / soln_ext[0];

    //Compute Roe averages
    dealii::Tensor<1,dim,real> vel_roe_avg;
    real vel_roe_avg_sqr = 0.0;
    real contravariant_vel = 0.0;
    real vel_R = 0.0;
    real vel_L = 0.0;
    for(int idim=0; idim<dim; idim++){
        vel_roe_avg[idim] = (soln_int[idim+1]/soln_int[0]*sqrt(soln_int[0])
                          + soln_ext[idim+1]/soln_ext[0]*sqrt(soln_ext[0]))
                          / (sqrt(soln_int[0]) + sqrt(soln_ext[0]));
        vel_roe_avg_sqr += vel_roe_avg[idim] * vel_roe_avg[idim];
        contravariant_vel += vel_roe_avg[idim] * normal_int[idim];
        vel_R += soln_ext[idim+1] / soln_ext[0] * normal_int[idim];
        vel_L += soln_int[idim+1] / soln_int[0] * normal_int[idim];
    }
    const real enthalpy_roe_avg = (specific_total_enthalpy_int*sqrt(soln_int[0])
                                + specific_total_enthalpy_ext*sqrt(soln_ext[0]))
                                / (sqrt(soln_int[0]) + sqrt(soln_ext[0]));
    const real speed_sound = sqrt(euler_physics->gamm1 * (enthalpy_roe_avg - 0.5 * vel_roe_avg_sqr));
    const real gamma_minus = (euler_physics->gam + 1.0) / (2.0*euler_physics->gam) * contravariant_vel
                           - sqrt( (euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) *(euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) + speed_sound * speed_sound / euler_physics->gam );
    const real gamma_plus = (euler_physics->gam + 1.0) / (2.0*euler_physics->gam) * contravariant_vel
                           + sqrt( (euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) *(euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) + speed_sound * speed_sound / euler_physics->gam );
    const real mach_number = contravariant_vel / speed_sound;
    real beta = 0.0;
    if(mach_number < 1.0 && mach_number >= 0.0) {
        real val = (contravariant_vel + gamma_minus) / (contravariant_vel - gamma_minus);
        if(val > 0.0)
            beta = val;
        else
            beta = 0.0;
    }
    else if(mach_number < 0.0 && mach_number >= -1){
        real val = (contravariant_vel + gamma_plus) / (contravariant_vel - gamma_plus);
        if(val > 0.0)
            beta = - val;
        else
            beta = 0.0;
    }
    else if(mach_number >= 1.0)
        beta = 1.0;
    else if(mach_number <= -1.0)
        beta = -1.0;

    real alpha_c = 0.0;
    if(abs(beta) <= 1e-14)
        alpha_c = abs(contravariant_vel);
    else if (beta > 0.0 && 0.0 < mach_number && mach_number < 1.0)
        alpha_c = - (1.0 + beta) * gamma_minus;
    else if (beta < 0.0 && -1.0 < mach_number && mach_number < 0.0)
        alpha_c =  (1.0 - beta) * gamma_plus;
    else if (abs(mach_number) >= 1.0)
        alpha_c = 0.0;

    std::array<real,nstate> dissipation;
    for(int istate=0;istate<nstate; istate++){
        const real u_L = (istate == nstate-1) ? soln_int[0] * specific_total_enthalpy_int
                  : soln_int[istate];
        const real u_R = (istate == nstate-1) ? soln_ext[0] * specific_total_enthalpy_ext
                  : soln_ext[istate];
        dissipation[istate] = - 0.5 * alpha_c * (u_R - u_L);
        dissipation[istate] -= 0.5 * beta *(u_R * vel_R - u_L * vel_L);
        if(istate > 0 && istate < nstate - 1){//momentum equations add pressure
            dissipation[istate] -= 0.5 * beta * (pressure_ext * normal_int[istate-1]
                                 - pressure_int * normal_int[istate-1]);
        }

    }

    return dissipation;
*/
}

// Instantiation
template class NumericalFluxConvective<PHILIP_DIM, 1, double>;
template class NumericalFluxConvective<PHILIP_DIM, 2, double>;
template class NumericalFluxConvective<PHILIP_DIM, 3, double>;
template class NumericalFluxConvective<PHILIP_DIM, 4, double>;
template class NumericalFluxConvective<PHILIP_DIM, 5, double>;
template class NumericalFluxConvective<PHILIP_DIM, 6, double>;
template class NumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, RadType >;

template class NumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class LaxFriedrichs<PHILIP_DIM, 1, double>;
template class LaxFriedrichs<PHILIP_DIM, 2, double>;
template class LaxFriedrichs<PHILIP_DIM, 3, double>;
template class LaxFriedrichs<PHILIP_DIM, 4, double>;
template class LaxFriedrichs<PHILIP_DIM, 5, double>;
template class LaxFriedrichs<PHILIP_DIM, 6, double>;
template class LaxFriedrichs<PHILIP_DIM, 1, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, RadFadType >;

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
template class HLLC<PHILIP_DIM, PHILIP_DIM+2, double>;
template class HLLC<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class HLLC<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class HLLC<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class HLLC<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class Central<PHILIP_DIM, 1, double>;
template class Central<PHILIP_DIM, 2, double>;
template class Central<PHILIP_DIM, 3, double>;
template class Central<PHILIP_DIM, 4, double>;
template class Central<PHILIP_DIM, 5, double>;
//template class Central<PHILIP_DIM, 6, double>;
template class Central<PHILIP_DIM, 1, FadType >;
template class Central<PHILIP_DIM, 2, FadType >;
template class Central<PHILIP_DIM, 3, FadType >;
template class Central<PHILIP_DIM, 4, FadType >;
template class Central<PHILIP_DIM, 5, FadType >;
//template class Central<PHILIP_DIM, 6, FadType >;
template class Central<PHILIP_DIM, 1, RadType >;
template class Central<PHILIP_DIM, 2, RadType >;
template class Central<PHILIP_DIM, 3, RadType >;
template class Central<PHILIP_DIM, 4, RadType >;
template class Central<PHILIP_DIM, 5, RadType >;
//template class Central<PHILIP_DIM, 6, RadType >;
template class Central<PHILIP_DIM, 1, FadFadType >;
template class Central<PHILIP_DIM, 2, FadFadType >;
template class Central<PHILIP_DIM, 3, FadFadType >;
template class Central<PHILIP_DIM, 4, FadFadType >;
template class Central<PHILIP_DIM, 5, FadFadType >;
//template class Central<PHILIP_DIM, 6, FadFadType >;
template class Central<PHILIP_DIM, 1, RadFadType >;
template class Central<PHILIP_DIM, 2, RadFadType >;
template class Central<PHILIP_DIM, 3, RadFadType >;
template class Central<PHILIP_DIM, 4, RadFadType >;
template class Central<PHILIP_DIM, 5, RadFadType >;
//template class Central<PHILIP_DIM, 6, RadFadType >;

template class EntropyConserving<PHILIP_DIM, 1, double>;
template class EntropyConserving<PHILIP_DIM, 2, double>;
template class EntropyConserving<PHILIP_DIM, 3, double>;
template class EntropyConserving<PHILIP_DIM, 4, double>;
template class EntropyConserving<PHILIP_DIM, 5, double>;
//template class EntropyConserving<PHILIP_DIM, 6, double>;
template class EntropyConserving<PHILIP_DIM, 1, FadType >;
template class EntropyConserving<PHILIP_DIM, 2, FadType >;
template class EntropyConserving<PHILIP_DIM, 3, FadType >;
template class EntropyConserving<PHILIP_DIM, 4, FadType >;
template class EntropyConserving<PHILIP_DIM, 5, FadType >;
//template class EntropyConserving<PHILIP_DIM, 6, FadType >;
template class EntropyConserving<PHILIP_DIM, 1, RadType >;
template class EntropyConserving<PHILIP_DIM, 2, RadType >;
template class EntropyConserving<PHILIP_DIM, 3, RadType >;
template class EntropyConserving<PHILIP_DIM, 4, RadType >;
template class EntropyConserving<PHILIP_DIM, 5, RadType >;
//template class EntropyConserving<PHILIP_DIM, 6, RadType >;
template class EntropyConserving<PHILIP_DIM, 1, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 2, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 3, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 4, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 5, FadFadType >;
//template class EntropyConserving<PHILIP_DIM, 6, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 1, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 2, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 3, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 4, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 5, RadFadType >;
//template class EntropyConserving<PHILIP_DIM, 6, RadFadType >;

template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, double>;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, FadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, RadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, FadFadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, RadFadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, RadFadType >;

template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, double>;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadType >;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadType >;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadFadType >;
//
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, double>;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadType >;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadType >;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadFadType >;

template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, double>;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class RiemannSolverDissipation<PHILIP_DIM, 1, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 2, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 3, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 4, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 5, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 6, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 1, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 1, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 1, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 1, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, RadFadType >;

template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, double>;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, FadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, RadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, FadFadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, RadFadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, RadFadType >;

template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, RadFadType >;

template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class HLLCBaselineNumericalFluxConvective<PHILIP_DIM, PHILIP_DIM+2, double>;
template class HLLCBaselineNumericalFluxConvective<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class HLLCBaselineNumericalFluxConvective<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class HLLCBaselineNumericalFluxConvective<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class HLLCBaselineNumericalFluxConvective<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
} // NumericalFlux namespace
} // PHiLiP namespace
