#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "amiet_model.hpp"

namespace PHiLiP {

//================================================================
// Amiet's model
//================================================================
template <int dim,int nstate,typename real,typename MeshType>
AmietModelFunctional<dim,nstate,real,MeshType>
::AmietModelFunctional(
    std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
    const ExtractionFunctional<dim,nstate,real,MeshType> & boundary_layer_extraction_input,
    const dealii::Point<3,real> & observer_coord_ref_input)
    : Functional<dim,nstate,real,MeshType>(dg_input)
    , acoustic_contribution_type(this->dg->all_parameters->amiet_param.acoustic_contribution_type)
    , wall_pressure_spectral_model_type(this->dg->all_parameters->amiet_param.wall_pressure_spectral_model_type)
    , boundary_layer_extraction(boundary_layer_extraction_input)
    , omega_min(this->dg->all_parameters->amiet_param.omega_min)
    , omega_max(this->dg->all_parameters->amiet_param.omega_max)
    , d_omega(this->dg->all_parameters->amiet_param.omega_interval)
    , numb_of_omega((omega_max-omega_min)/d_omega)
    , observer_coord_ref(observer_coord_ref_input)
    , R_specific(this->dg->all_parameters->amiet_param.R_specific)
    , ref_density(this->dg->all_parameters->amiet_param.ref_density)
    , ref_length(this->dg->all_parameters->euler_param.ref_length)
    , ref_temperature(this->dg->all_parameters->navier_stokes_param.temperature_inf)
    , mach_inf(this->dg->all_parameters->euler_param.mach_inf)
    , sound_inf(sqrt(this->dg->all_parameters->euler_param.gamma_gas*R_specific*ref_temperature))
    , ref_U(mach_inf*sound_inf)
    , ref_viscosity(ref_density*ref_U*ref_length/this->dg->all_parameters->navier_stokes_param.reynolds_number_inf)
    , U_inf(this->boundary_layer_extraction.U_inf*ref_U)
    , density_inf(this->boundary_layer_extraction.density_inf*ref_density)
    , chord_length(this->dg->all_parameters->amiet_param.chord_length*ref_length)
    , span_length(this->dg->all_parameters->amiet_param.span_length*ref_length)
    , alpha(this->dg->all_parameters->amiet_param.alpha)
    , U_c(U_inf/alpha)
    , U_edge(this->boundary_layer_extraction.evaluate_edge_velocity()*ref_U)
    , friction_velocity(this->boundary_layer_extraction.evaluate_friction_velocity()*ref_U)
    , boundary_layer_thickness(this->boundary_layer_extraction.evaluate_boundary_layer_thickness()*ref_length)
    , displacement_thickness(this->boundary_layer_extraction.evaluate_displacement_thickness()*ref_length)
    , momentum_thickness(this->boundary_layer_extraction.evaluate_momentum_thickness()*ref_length)
    , wall_shear_stress(this->boundary_layer_extraction.evaluate_wall_shear_stress()*ref_density*ref_U*ref_U)
    , maximum_shear_stress(this->boundary_layer_extraction.evaluate_maximum_shear_stress()*ref_density*ref_U*ref_U)
    , kinematic_viscosity(ref_viscosity/ref_density)
    , pressure_gradient_tangential(std::abs(this->boundary_layer_extraction.evaluate_pressure_gradient_tangential())*ref_density*ref_U*ref_U/ref_length)
    , clauser_equilibrium_parameter(momentum_thickness/wall_shear_stress*pressure_gradient_tangential)
    , cole_wake_parameter(0.8*pow(clauser_equilibrium_parameter+0.5,3.0/4.0))
    , zagarola_smits_parameter(boundary_layer_thickness/displacement_thickness)
    , b(chord_length/2.0)
    , beta_sqr(1.0-mach_inf*mach_inf)
    , S0(sqrt(observer_coord_ref[0]*observer_coord_ref[0]+beta_sqr*(observer_coord_ref[1]*observer_coord_ref[1]+observer_coord_ref[2]*observer_coord_ref[2])))
{
    std::complex<real> imag (0.0,1.0);
    imag_unit = imag;
 
    std::cout << "ref_density is "                   << ref_density << std::endl;
    std::cout << "ref_temperature is "               << ref_temperature << std::endl;
    std::cout << "ref_U is "                         << ref_U << std::endl;
    std::cout << "ref_viscosity is "                 << ref_viscosity << std::endl;

    std::cout << "U_inf is "                         << U_inf << std::endl;
    std::cout << "sound_inf is "                     << sound_inf << std::endl;
    std::cout << "density_inf is "                   << density_inf << std::endl;
    std::cout << "U_c is "                           << U_c << std::endl;
    std::cout << "U_edge is "                        << U_edge << std::endl;
    std::cout << "friction_velocity is "             << friction_velocity << std::endl;
    std::cout << "boundary_layer_thickness is "      << boundary_layer_thickness << std::endl;
    std::cout << "displacement_thickness is "        << displacement_thickness << std::endl;
    std::cout << "momentum_thickness is "            << momentum_thickness << std::endl;
    std::cout << "wall_shear_stress is "             << wall_shear_stress << std::endl;
    std::cout << "maximum_shear_stress is "          << maximum_shear_stress << std::endl;
    std::cout << "kinematic_viscosity is "           << kinematic_viscosity << std::endl;
    std::cout << "pressure_gradient_tangential is "  << pressure_gradient_tangential << std::endl;
    std::cout << "clauser_equilibrium_parameter is " << clauser_equilibrium_parameter << std::endl;
    std::cout << "cole_wake_parameter is "           << cole_wake_parameter << std::endl;
    std::cout << "zagarola_smits_parameter is "      << zagarola_smits_parameter << std::endl;

    Phi_pp.resize(numb_of_omega);
    S_pp.resize(numb_of_omega);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_functional(const bool compute_dIdW, 
                      const bool compute_dIdX, 
                      const bool compute_d2I)
{
    double value = Functional<dim,nstate,real,MeshType>::evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);

    return value;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD(const real omega) const
{
    switch(this->wall_pressure_spectral_model_type) {
        case Wall_Pressure_Spectral_Model_types::Goody : 
            return this->wall_pressure_PSD_Goody(omega);
            break;
        case Wall_Pressure_Spectral_Model_types::Rozenberg : 
            return this->wall_pressure_PSD_Rozenburg(omega);
            break;
        case Wall_Pressure_Spectral_Model_types::Kamruzzaman : 
            return this->wall_pressure_PSD_Kamruzzaman(omega);
            break;
        default: 
            break;
    }
    std::cout << "ERROR: Fail to determine wall pressure spectrum model type for Amiet's model..." << std::endl;
    std::abort();
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD_Goody(const real omega) const
{
    // Goody's model for wall pressure spectrum
    const real a = 3.0;
    const real b = 2.0;
    const real c = 0.75;
    const real d = 0.5;
    const real e = 3.7;
    const real f = 1.1;
    const real g = -0.57;
    const real h = 7.0;
    const real i = 1.0;

    const real R_T = (boundary_layer_thickness/U_edge)/(kinematic_viscosity/(friction_velocity*friction_velocity));
    const real Phi_star = wall_shear_stress*wall_shear_stress*boundary_layer_thickness/U_edge;
    const real omega_star = omega*boundary_layer_thickness/U_edge;

    real Phi_pp = Phi_star*a*pow(omega_star,b)/(pow(i*pow(omega_star,c)+d,e)+pow(f*pow(R_T,g)*omega_star,h));

    return Phi_pp;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD_Rozenburg(const real omega) const
{
    // Rozenburg's model for wall pressure spectrum
    const real Pi = cole_wake_parameter;
    const real Delta_star = zagarola_smits_parameter;
    const real b = 2.0;
    const real c = 0.75;
    const real e = 3.7+1.5*clauser_equilibrium_parameter;
    const real d = 4.76*pow(1.4/Delta_star,0.75)*(0.375*e-1.0);
    const real a = 2.82*Delta_star*Delta_star*pow(6.13*pow(Delta_star,-0.75)+d,e)*(4.2*Pi/Delta_star+1.0);
    const real f = 8.8;
    const real g = -0.57;
    const real i = 4.76;
    const real R_T = (boundary_layer_thickness/U_edge)/(kinematic_viscosity/(friction_velocity*friction_velocity));

    const real const_l = 3.0;
    const real const_r = 19.0/sqrt(R_T);
#if const_l<const_r
    const real h = 3.0+7.0;
#else
    const real h =19.0/sqrt(R_T)+7.0 ;
#endif
    (void) const_l;
    (void) const_r;
  
    const real Phi_star = maximum_shear_stress*maximum_shear_stress*displacement_thickness/U_edge;
    const real omega_star = omega*displacement_thickness/U_edge;

    real Phi_pp = Phi_star*a*pow(omega_star,b)/(pow(i*pow(omega_star,c)+d,e)+pow(f*pow(R_T,g)*omega_star,h));

    return Phi_pp;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD_Kamruzzaman(const real omega) const
{
    // Kamruzzaman's model for wall pressure spectrum
    const real Pi = cole_wake_parameter;
    const real G = 6.1*sqrt(clauser_equilibrium_parameter+1.81)-1.7;
    const real C_f = wall_shear_stress/(0.5*density_inf*U_inf*U_inf);
    const real lambda = sqrt(2.0/C_f);
    const real H = 1.0-G/lambda; 
    const real m = 0.5*pow(H/1.31,0.3);
    const real a = 0.45*(1.75*pow(Pi*Pi*clauser_equilibrium_parameter*clauser_equilibrium_parameter,m)+15.0);
    const real b = 2.0;
    const real c = 1.637;
    const real d = 0.27;
    const real e = 2.47;
    const real f = pow(1.15,-2.0/7.0);
    const real g = -2.0/7.0;
    const real h = 7.0;
    const real i = 1.0;

    const real R_T = (displacement_thickness/U_edge)/(kinematic_viscosity/(friction_velocity*friction_velocity));
    const real Phi_star = wall_shear_stress*wall_shear_stress*displacement_thickness/U_edge;
    const real omega_star = omega*displacement_thickness/U_edge;

    real Phi_pp = Phi_star*a*pow(omega_star,b)/(pow(i*pow(omega_star,c)+d,e)+pow(f*pow(R_T,g)*omega_star,h));

    return Phi_pp;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::spanwise_correlation_length(const real omega) const
{
    // Corcos model for spanwise correlation length
    const real b_c = 1.47;
    return b_c*U_c/omega;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::E(const std::complex<real> z) const
{
    return (1.0+imag_unit)/2.0*Faddeeva::erf(sqrt(-imag_unit*z));
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::E_star(const std::complex<real> z) const
{
    return (1.0-imag_unit)/2.0*Faddeeva::erf(sqrt(imag_unit*z));
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::ES_star(const std::complex<real> z) const
{
    std::complex<real> ES_star;
    if (z.real()==0.0 && z.imag()==0.0){
        ES_star = sqrt(2.0/pi);
    } else {
        ES_star = E_star(z)/sqrt(z);
    }
    if (z.real()<0.0 && z.imag()>=0.0) {
        ES_star *= -1.0;
    }
    return ES_star;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::radiation_integral_trailing_edge_main (const real B,
                                         const real C,
                                         const real mu_bar,
                                         const real S0,
                                         const real kappa_bar_prime,
                                         const std::complex<real> A1_prime,
                                         const bool is_supercritical) const
{
    std::complex<real> radiation_integral_trailing_edge_main;
    
    if (is_supercritical) {
        radiation_integral_trailing_edge_main 
            = -exp(2.0*imag_unit*C)/(imag_unit*C)*((1.0+imag_unit)*exp(-2.0*imag_unit*C)*sqrt(2.0*B)*ES_star(2.0*(B-C))-(1.0+imag_unit)*E_star(2.0*B)+1.0-exp(-2.0*imag_unit*C));
    } else {
        radiation_integral_trailing_edge_main 
            = -exp(2.0*imag_unit*C)/(imag_unit*C)*(exp(-2.0*imag_unit*C)*sqrt(2.0*A1_prime)*(1.0+imag_unit)*ES_star(2.0*(mu_bar*observer_coord_ref[0]/S0-imag_unit*kappa_bar_prime))-Faddeeva::erf(sqrt(2.0*imag_unit*A1_prime))+1.0);
    }
    
    return radiation_integral_trailing_edge_main;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::radiation_integral_trailing_edge_back (const real C,
                                         const real D,
                                         const real kappa_bar,
                                         const real kappa_bar_prime,
                                         const real K_bar,
                                         const real mu_bar,
                                         const real S0,
                                         const real mach_inf,
                                         const std::complex<real> A_prime,
                                         const std::complex<real> G,
                                         const std::complex<real> D_prime,
                                         const std::complex<real> H,
                                         const std::complex<real> H_prime,
                                         const bool is_supercritical) const 
{
    std::complex<real> radiation_integral_trailing_edge_back;

    if (is_supercritical) {
      radiation_integral_trailing_edge_back
          = H*(pow(exp(4.0*imag_unit*kappa_bar)*(1.0-(1.0+imag_unit)*E_star(4.0*kappa_bar)),C)-exp(2.0*imag_unit*D)+imag_unit*(D+K_bar+mach_inf*mu_bar-kappa_bar)*G);
    } else {
      radiation_integral_trailing_edge_back
          = exp(-2.0*imag_unit*D_prime)/D_prime*H_prime*(A_prime*(exp(2.0*imag_unit*D_prime)*(1.0-Faddeeva::erf(sqrt(4.0*kappa_bar_prime)))-1.0)+2.0*sqrt(2.0*kappa_bar_prime)*(K_bar+(mach_inf-observer_coord_ref[0]/S0)*mu_bar)*ES_star(-2.0*D_prime));
    }
    
    return radiation_integral_trailing_edge_back;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::radiation_integral_trailing_edge (const real omega) const
{
    const real k = omega/sound_inf;
    const real K = k/mach_inf;
    const real K_bar = K*b;
    const real K1_bar = K_bar*alpha;
    const real K2_bar = k*observer_coord_ref[1]/S0*b;
    const real mu_bar = K_bar*mach_inf/beta_sqr;
    const real kappa_bar = sqrt(mu_bar*mu_bar-K2_bar*K2_bar/beta_sqr);
    const real kappa_bar_prime = sqrt(K2_bar*K2_bar/beta_sqr-mu_bar*mu_bar);
    const real epsilon = pow(1.0+1.0/(4.0*mu_bar),-1.0/2.0);
    
    const real Theta = sqrt((K1_bar+mach_inf*mu_bar+kappa_bar)/(K_bar+mach_inf*mu_bar+kappa_bar));
    const std::complex<real> A_prime = K_bar+mach_inf*mu_bar-kappa_bar_prime*imag_unit;
    const std::complex<real> A1_prime = K1_bar+mach_inf*mu_bar-kappa_bar_prime*imag_unit;
    const std::complex<real> Theta_prime = sqrt(A1_prime/A_prime);
    
    const real B = K1_bar+mach_inf*mu_bar+kappa_bar;
    const real C = K1_bar-mu_bar*(observer_coord_ref[0]/S0-mach_inf);
    const real D = kappa_bar-mu_bar*observer_coord_ref[0]/S0;
    const std::complex<real> D_prime = mu_bar*observer_coord_ref[0]/S0-kappa_bar_prime*imag_unit;
    const std::complex<real> 
        G = (1.0+epsilon)*exp(imag_unit*(2.0*kappa_bar+D))*sin(D-2.0*kappa_bar)/(D-2.0*kappa_bar)
            +(1.0-epsilon)*exp(imag_unit*(-2.0*kappa_bar+D))*sin(D+2.0*kappa_bar)/(D+2.0*kappa_bar)
            +(1.0+epsilon)*(1.0-imag_unit)/(2.0*(D-2.0*kappa_bar))*exp(4.0*kappa_bar*imag_unit)*E_star(4.0*kappa_bar)
            -(1.0-epsilon)*(1.0+imag_unit)/(2.0*(D+2.0*kappa_bar))*exp(-4.0*kappa_bar*imag_unit)*E(4.0*kappa_bar)
            +exp(2.0*D*imag_unit)/sqrt(2.0)*sqrt((2.0*kappa_bar)/D)*E_star(2.0*D)*((1.0+imag_unit)*(1.0-epsilon)/(D+2.0*kappa_bar)-(1.0-imag_unit)*(1.0+epsilon)/(D-2.0*kappa_bar));
    const std::complex<real> 
        H = (1.0+imag_unit)*exp(-4.0*kappa_bar*imag_unit)*(1.0-Theta*Theta)/(2.0*sqrt(pi)*(alpha-1.0)*K_bar*sqrt(B));
    const std::complex<real>
        H_prime = (1.0+imag_unit)*(1.0-Theta_prime*Theta_prime)/(2.0*sqrt(pi)*(alpha-1.0)*K_bar*sqrt(A1_prime));

    const bool is_supercritical = K2_bar*K2_bar < mu_bar*mu_bar*beta_sqr;

    std::complex<real> RI_TE_main;
    std::complex<real> RI_TE_back;
    switch(this->acoustic_contribution_type) {
        case Acoustic_Contribution_types::main : 
            RI_TE_main = radiation_integral_trailing_edge_main(B,C,mu_bar,S0,kappa_bar_prime,A1_prime,is_supercritical);
            return RI_TE_main;
            break;
        case Acoustic_Contribution_types::back : 
            RI_TE_back = radiation_integral_trailing_edge_back(C,D,kappa_bar,kappa_bar_prime,K_bar,mu_bar,S0,mach_inf,A_prime,G,D_prime,H,H_prime,is_supercritical);
            return RI_TE_back;
            break;
        case Acoustic_Contribution_types::main_and_back : 
            RI_TE_main = radiation_integral_trailing_edge_main(B,C,mu_bar,S0,kappa_bar_prime,A1_prime,is_supercritical);
            RI_TE_back = radiation_integral_trailing_edge_back(C,D,kappa_bar,kappa_bar_prime,K_bar,mu_bar,S0,mach_inf,A_prime,G,D_prime,H,H_prime,is_supercritical);
            return RI_TE_main+RI_TE_back;
            break;
        default: 
            break;
    }
    std::cout << "ERROR: Fail to determine contribution type for radiation integral..." << std::endl;
    std::abort();
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::acoustic_PSD(const real omega,
               const real Phi_pp_of_sampling) const
{
    const std::complex<real> radiation_integral = radiation_integral_trailing_edge(omega);
    const real l_y = spanwise_correlation_length(omega);
 
    real S_pp_of_sampling;

    S_pp_of_sampling = pow(omega*observer_coord_ref[2]*b/(2.0*pi*sound_inf*S0*S0),2.0)*2.0*span_length*pow(abs(radiation_integral),2.0)*Phi_pp_of_sampling*l_y;

    return S_pp_of_sampling;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
void AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_wall_pressure_acoustic_spectrum()
{   
    for (int i=0;i<numb_of_omega;++i){
        const real omega_of_sampling = omega_min+i*d_omega;
        Phi_pp[i] = wall_pressure_PSD(omega_of_sampling);
        S_pp[i] = acoustic_PSD(omega_of_sampling,Phi_pp[i]);
    }
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
void AmietModelFunctional<dim,nstate,real,MeshType>
::output_wall_pressure_acoustic_spectrum_dat()
{   
    std::ofstream outfile_S_pp_Phi_pp;
    outfile_S_pp_Phi_pp.open("S_pp_and_Phi_pp.dat");
    for(int i=0;i<numb_of_omega;++i){
        const real omega_of_sampling = omega_min+i*d_omega;
        outfile_S_pp_Phi_pp << omega_of_sampling/(2.0*pi) << "\t\t" 
                            << 10.0*log10(8.0*pi*S_pp[i]/(2e-5*2e-5)) << "\t\t" 
                            << 10.0*log10(2.0*pi*Phi_pp[i]/(2e-5*2e-5)) << "\n";
    }
    outfile_S_pp_Phi_pp.close();
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- AmietModelFunctional
#if PHILIP_DIM!=1
template class AmietModelFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AmietModelFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace