#ifndef __AMIET_MODEL_H__
#define __AMIET_MODEL_H__

#include <complex>
#include <cmath>
#include "Faddeeva.hh"
#include "functional.h"
#include "extraction_functional.hpp"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
class AmietModelFunctional : public Functional<dim, nstate, real, MeshType>
{
private:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real,MeshType>::evaluate_volume_integrand;

    using Acoustic_Contribution_types = Parameters::AmietParam::AcousticContributionEnum;
    using Wall_Pressure_Spectral_Model_types = Parameters::AmietParam::WallPressureSpectralModelEnum;

    /// @brief Switches between leading and trailing edge scattering types.
    const Acoustic_Contribution_types acoustic_contribution_type;

    /// @brief Switches between Goody, Rozenberg and Kamruzzaman wall pressure spectral model types.
    const Wall_Pressure_Spectral_Model_types wall_pressure_spectral_model_type;

    /// @brief Constant pi = 3.1415926...
    const real pi = atan(1)*4.0;

    /// @brief Imaginary unit i of complex number
    std::complex<real> imag_unit;

    /// @brief ExtractionFunctional Object provides all necessary boundary parameters
    const ExtractionFunctional<dim,nstate,real,MeshType> boundary_layer_extraction;

    /// @brief Lower limit of investigated frequency (rad/s).
    const real omega_min;
    /// @brief Upper limit of investigated frequency (rad/s).
    const real omega_max;
    /// @brief Interval of investigated frequency (rad/s).
    const real d_omega;
    /// @brief Number of sampling frequency.
    const int numb_of_omega;

    /// @brief Coordinate of observer in airfoil frame of reference (coordinate origin located at the center of trailing edge).
    /** Observer coordinate is based on the reference coordinate built at the center of trailing edge as sketched                                    
     *                       ------------------
     *              /       /                /     z
     *             /       /                /      ^
     *            /       /                /       |
     *      b <- /       /--------------- /(0,0,0) ----> x 
     *          /       /       /        /        /
     *         /       /       /-> b/2  /        v 
     *        /       /       /        /        y
     *               ------------------
     *  Note: the observer coordinate is most likely different from the one used for flow solver 
     */
    const dealii::Point<3,real> observer_coord_ref;

    /// @brief Specific gas constant R. Units: [J/(kg*K)].
    const real R_specific;
    /// @brief Reference density. Units: [kg/m^3].
    const real ref_density;
    /// @brief Reference length. Units: [m].
    const real ref_length;
    /// @brief Reference temperature. Units: [K].
    const real ref_temperature;
    /// @brief Mach number of free stream.
    const real mach_inf;
    /// @brief Dimensionalized sound speed of free stream. Units: [m/s].
    const real sound_inf;
    /// @brief Reference flow speed. Units: [m/s].
    const real ref_speed;
    /// @brief Reference dynamic viscosity. Units: [kg/(m*s)].
    const real ref_viscosity;
    /// @brief Dimensionalized flow speed of free stream. Units: [m/s].
    //const real U_inf;
    /// @brief Dimensionalized density of free stream. Units: [kg/m^3].
    //const real density_inf;

    /// @brief Dimensionalized chord length of airfoil. Units: [m].
    const real chord_length;
    /// @brief Dimensionalized span length of airfoil. Units: [m].
    const real span_length;

    /// @brief Ratio of free-stream and convection speed of turbulence, alpha = U_inf/U_c.
    const real alpha;
    /// @brief Dimensionalized convection velocity of turbulence. Units: [m/s].
    //const real U_c;
    /// @brief Dimensionalized edge velocity of boundary layer. Units: [m/s].
    //const real U_edge;
    /// @brief Dimensionalized friction velocity of boundary layer. Units: [m/s].
    //const real friction_velocity;
    /// @brief Dimensionalized boundary layer thickness. Units: [m].
    //const real boundary_layer_thickness;
    /// @brief Dimensionalized displacement thickness. Units: [m].
    //const real displacement_thickness;
    /// @brief Dimensionalized momentum thickness. Units: [m].
    //const real momentum_thickness;
    /// @brief Dimensionalized wall shear stress. Units: [Pa].
    //const real wall_shear_stress;
    /// @brief Dimensionalized maximum shear stress over extraction line. Units: [Pa].
    //const real maximum_shear_stress;
    /// @brief Dimensionalized kinematic viscosity. Units: [m^2/s].
    //const real kinematic_viscosity;
    /// @brief Dimensionalized pressure gradient over stream wise direction. Units: [Pa/m].
    //const real pressure_gradient_tangential;

    /// @brief Clauser's equilibrium parameter.
    //const real clauser_equilibrium_parameter;
    /// @brief Cole's wake parameter.
    //const real cole_wake_parameter;
    /// @brief Zagarola-Smits's parameter.
    //const real zagarola_smits_parameter;

    /// @brief Half of chord length
    const real b;
    /// @brief beta^2 = 1-Mach_inf^2
    const real beta_sqr;
    /// @brief S0 = sqrt(x^2+beta^2*(y^2+z^2))
    const real S0;

    /// @brief Vector of wall-pressure spectrum over investigated frequency range
    //std::vector<real> Phi_pp;
    /// @brief Vector of far-field acoustic spectrum over investigated frequency range
    //std::vector<real> S_pp;

    //dealii::LinearAlgebra::distributed::Vector<FadFadType> solution_fad_fad;

    //dealii::LinearAlgebra::distributed::Vector<FadFadType> volume_nodes_fad_fad;

public:
    /// Constructor
    AmietModelFunctional(
        std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
        const ExtractionFunctional<dim,nstate,real,MeshType> & extraction_input,
        const dealii::Point<3,real> & observer_coord_ref_input);
    /// Destructor
    ~AmietModelFunctional(){};

    /// Allocate and setup the derivative dW_int_dW vector.
    //void allocate_dW_int_dW();

    real evaluate_functional(
        const bool compute_dIdW = false, 
        const bool compute_dIdX = false, 
        const bool compute_d2I = false) override;

    /// Function to evaluate wall-pressure power spectral density, Phi_pp, for a given frequency, omega.
    template <typename real2>
    real2 wall_pressure_PSD(
        const real omega,
        const real speed_free_stream,
        const real density_free_stream,
        const real edge_velocity,
        const real boundary_layer_thickness,
        const real maximum_shear_stress,
        const real2 displacement_thickness,
        const real2 momentum_thickness,
        const real2 friction_velocity,
        const real2 wall_shear_stress,
        const real2 pressure_gradient_tangential,
        const real2 kinematic_viscosity) const;

    /// Function to evaluate wall-pressure power spectral density using Goody's model.
    template <typename real2>
    real2 wall_pressure_PSD_Goody(
        const real omega,
        const real edge_velocity,
        const real boundary_layer_thickness,
        const real2 friction_velocity,
        const real2 wall_shear_stress,
        const real2 kinematic_viscosity) const;

    /// Function to evaluate wall-pressure power spectral density using Rozenburg's model.
    template <typename real2>
    real2 wall_pressure_PSD_Rozenburg(
        const real omega,
        const real edge_velocity,
        const real boundary_layer_thickness,
        const real maximum_shear_stress,
        const real2 displacement_thickness,
        const real2 momentum_thickness,
        const real2 friction_velocity,
        const real2 wall_shear_stress,
        const real2 pressure_gradient_tangential,
        const real2 kinematic_viscosity) const;

    /// Function to evaluate wall-pressure power spectral density using Kamruzzaman's model.
    template <typename real2>
    real2 wall_pressure_PSD_Kamruzzaman(
        const real omega,
        const real speed_free_stream,
        const real density_free_stream,
        const real edge_velocity,
        const real2 displacement_thickness,
        const real2 momentum_thickness,
        const real2 friction_velocity,
        const real2 wall_shear_stress,
        const real2 pressure_gradient_tangential,
        const real2 kinematic_viscosity) const;

    template <typename real2>
    real2 evaluate_time_scale_ratio(
        const real boundary_layer_thickness,
        const real edge_velocity,
        const real2 friction_velocity,
        const real2 kinematic_viscosity) const;

    template <typename real2>
    real2 evaluate_clauser_equilibrium_parameter(
        const real2 momentum_thickness,
        const real2 wall_shear_stress,
        const real2 pressure_gradient_tangential) const;

    template <typename real2>
    real2 evaluate_cole_wake_parameter(
        const real2 momentum_thickness,
        const real2 wall_shear_stress,
        const real2 pressure_gradient_tangential) const;

    template <typename real2>
    real2 evaluate_zagarola_smits_parameter(
        const real boundary_layer_thickness,
        const real2 displacement_thickness) const;

    /// Function to evaluate spanwise correlation length using Corcos's model.
    real spanwise_correlation_length(
        const real omega,
        const real U_c) const;

    /// Function to evaluate complex function of Fresnel integral
    std::complex<real> E (const std::complex<real> z) const;

    /// Function to evaluate complex function of Fresnel integral
    std::complex<real> E_star(const std::complex<real> z) const;

    /// Function to evaluate new complex function of Fresnel integral
    std::complex<real> ES_star (const std::complex<real> z) const;

    /// Function to evaluate radiation integral involving main contribution for supercritical or subcritical gust
    std::complex<real> radiation_integral_trailing_edge_main (
        const real B,
        const real C,
        const real mu_bar,
        const real S0,
        const real kappa_bar_prime,
        const std::complex<real> A1_prime,
        const bool is_supercritical) const;

    /// Function to evaluate radiation integral involving back-scattering correction for supercritical or subcritical gust
    std::complex<real> radiation_integral_trailing_edge_back (
        const real C,
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
        const bool is_supercritical) const;

    /// Function to evaluate radiation integral involving main contribution and/or back-scattering correction for supercritical or subcritical gust
    std::complex<real> radiation_integral_trailing_edge (const real omega) const;

    /// Function to evaluate far-field acoustic power spectral density, S_pp, for a given frequency, omega.
    template <typename real2>
    real2 acoustic_PSD(
        const real omega,
        const real U_c,
        const real2 Phi_pp_of_sampling) const;

    /// Function to evaluate vector of wall-pressure spectrum and far-field acoustic spectrum over investigated frequency range
    //void evaluate_wall_pressure_acoustic_spectrum();

    /// Function to evaluate overall far-field sound pressure level (OASPL) over investigated frequency range
    template <typename real2>
    real2 evaluate_overall_sound_pressure_level(const std::vector<real2> S_pp);

    /// Function to output vector of wall-pressure spectrum and far-field acoustic spectrum over investigated frequency range in a .dat file
    template <typename real2>
    void output_wall_pressure_acoustic_spectrum_dat(
        const std::vector<real2> &Phi_pp,
        const std::vector<real2> &S_pp);

};
} // PHiLiP namespace

#endif