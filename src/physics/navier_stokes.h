#ifndef __NAVIER_STOKES__
#define __NAVIER_STOKES__

#include "euler.h"
#include "parameters/parameters_navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Navier-Stokes equations. Derived from Euler for the convective terms, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class NavierStokes : public Euler <dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following:
    */
    using PhysicsBase<dim,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nstate,real>::source_term;
    using PhysicsBase<dim,nstate,real>::boundary_face_values;
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    NavierStokes( 
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const bool                                                use_constant_viscosity,
        const double                                              constant_viscosity,
        const double                                              temperature_inf = 273.15,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG,
        const bool                                                has_nonzero_physical_source = false);

    /// Nondimensionalized viscosity coefficient at infinity.
    const double viscosity_coefficient_inf;
    /// Flag to use constant viscosity instead of Sutherland's law of viscosity
    const bool use_constant_viscosity;
    /// Nondimensionalized constant viscosity
    const double constant_viscosity;
    /// Prandtl number
    const double prandtl_number;
    /// Farfield (free stream) Reynolds number
    const double reynolds_number_inf;
    /// Nondimensionalized isothermal wall temperature
    const double isothermal_wall_temperature;
    /// Thermal boundary condition type (adiabatic or isothermal)
    const thermal_boundary_condition_enum thermal_boundary_condition_type;

protected:    
    ///@{
    /** Constants for Sutherland's law for viscosity
     *  Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     *  Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const double sutherlands_temperature; ///< Sutherland's temperature. Units: [K]
    const double freestream_temperature; ///< Freestream temperature. Units: [K]
    const double temperature_ratio; ///< Ratio of Sutherland's temperature to freestream temperature
    //@}

public:
    /// Destructor
    ~NavierStokes() {};

    /** Nondimensionalized temperature gradient */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_temperature_gradient (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized velocities parallel to wall */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_velocities_parallel_to_wall(
        const std::array<real2,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real2> &normal_vector) const;

    /** Nondimensionalized wall tangent vector */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_wall_tangent_vector(
        const std::array<real2,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real2> &normal_vector) const;

    /** Nondimensionalized wall tangent vector from velocities parallel to wall */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_wall_tangent_vector_from_velocities_parallel_to_wall(
        const dealii::Tensor<1,dim,real2> &velocities_parallel_to_wall) const;

    /** Nondimensionalized wall shear stress */
    template<typename real2>
    real2 compute_wall_shear_stress (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient,
        const dealii::Tensor<1,dim,real2> &normal_vector) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Based on the use_constant_viscosity flag, it returns a value based on either:
     *  (1) Sutherland's viscosity law, or
     *  (2) Constant nondimensionalized viscosity value
     */
    template<typename real2>
    real2 compute_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Based on the use_constant_viscosity flag, it returns a value based on either:
     *  (1) Sutherland's viscosity law, or
     *  (2) Constant nondimensionalized viscosity value
     */
    template<typename real2>
    real2 compute_viscosity_coefficient_from_temperature (const real2 temperature) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     *  Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    template<typename real2>
    real2 compute_viscosity_coefficient_sutherlands_law (const std::array<real2,nstate> &primitive_soln) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     *  Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    template<typename real2>
    real2 compute_viscosity_coefficient_sutherlands_law_from_temperature (const real2 temperature) const;

    /** Scaled nondimensionalized viscosity coefficient, hat{mu*}, given nondimensionalized viscosity coefficient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    template<typename real2>
    real2 scale_viscosity_coefficient (const real2 viscosity_coefficient) const;

    /** Scaled nondimensionalized viscosity coefficient, hat{mu*} 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    template<typename real2>
    real2 compute_scaled_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized heat conductivity, hat{kappa*}, given scaled nondimensionalized viscosity coefficient and Prandtl number
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    template<typename real2>
    real2 compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number (
        const real2 scaled_viscosity_coefficient, 
        const double prandtl_number_input) const;

    /** Scaled nondimensionalized heat conductivity, hat{kappa*}
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    template<typename real2>
    real2 compute_scaled_heat_conductivity (const std::array<real2,nstate> &primitive_soln) const;

    /** Nondimensionalized heat flux, q*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_heat_flux (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized heat flux, q*, given the scaled heat conductivity and temperature gradient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient (
        const real2 scaled_heat_conductivity,
        const dealii::Tensor<1,dim,real2> &temperature_gradient) const;

    /// Evaluate vorticity from conservative variables and gradient of conservative variables
    template<typename real2>
    dealii::Tensor<1,3,real2> compute_vorticity (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const;

    /// Evaluate vorticity magnitude squared from conservative variables and gradient of conservative variables
    real compute_vorticity_magnitude_sqr (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /// Evaluate vorticity magnitude from conservative variables and gradient of conservative variables
    real compute_vorticity_magnitude (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /// Evaluate enstrophy from conservative variables and gradient of conservative variables
    real compute_enstrophy (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /// Evaluate incompressible enstrophy from conservative variables and gradient of conservative variables
    real compute_incompressible_enstrophy (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /// Evaluate incompressible palinstrophy from conservative variables and gradient of vorticity
    real compute_incompressible_palinstrophy (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,3> &vorticity_gradient) const;

    /** Evaluate non-dimensional theoretical vorticity-based dissipation rate integrated enstrophy. 
     *  Note: For incompressible flows or when dilatation effects are negligible
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     *  -- Equation (56) with free-stream nondimensionalization applied
     * */
    real compute_vorticity_based_dissipation_rate_from_integrated_enstrophy (
        const real integrated_enstrophy) const;

    /** Evaluate pressure dilatation from conservative variables and gradient of conservative variables
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    real compute_pressure_dilatation (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Evaluate the deviatoric strain-rate tensor from conservative variables and gradient of conservative variables
     *  -- Reference: de la Llave Plata et al. (2019). "On the performance of a high-order multiscale DG approach to LES at increasing Reynolds number."
     * */
    dealii::Tensor<2,dim,real> compute_deviatoric_strain_rate_tensor (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized strain rate tensor, S*, from conservative solution and solution gradient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, extracted from eq.(4.14.12)
     */
    dealii::Tensor<2,dim,real> compute_strain_rate_tensor_from_conservative (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized strain rate tensor, S*, from conservative solution and solution gradient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, extracted from eq.(4.14.12)
     */
    template<typename real2>
    dealii::Tensor<2,dim,real2> compute_strain_rate_tensor_from_conservative_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const;

    /// Evaluate the square of the deviatoric strain-rate tensor magnitude (i.e. double dot product) from conservative variables and gradient of conservative variables
    real compute_deviatoric_strain_rate_tensor_magnitude_sqr (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Evaluate non-dimensional theoretical deviatoric strain-rate tensor based dissipation rate from integrated
     *  deviatoric strain-rate tensor magnitude squared.
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     *  -- Equation (57a) with free-stream nondimensionalization applied
     * */
    real compute_deviatoric_strain_rate_tensor_based_dissipation_rate_from_integrated_deviatoric_strain_rate_tensor_magnitude_sqr (
        const real integrated_deviatoric_strain_rate_tensor_magnitude_sqr) const;

    /** Extract gradient of velocities */
    template<typename real2>
    dealii::Tensor<2,dim,real2> 
    extract_velocities_gradient_from_primitive_solution_gradient (
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized strain rate tensor, S*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, extracted from eq.(4.14.12)
     */
    template<typename real2>
    dealii::Tensor<2,dim,real2> 
    compute_strain_rate_tensor (
        const dealii::Tensor<2,dim,real2> &vel_gradient) const;

    /// Evaluate the square of the strain-rate tensor magnitude (i.e. double dot product) from conservative variables and gradient of conservative variables
    real compute_strain_rate_tensor_magnitude_sqr (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Evaluate non-dimensional theoretical strain-rate tensor based dissipation rate from integrated
     *  strain-rate tensor magnitude squared.
     *  -- Reference: Navah, Farshad, et al. "A High-Order Variational Multiscale Approach 
     *                to Turbulence for Compact Nodal Schemes." 
     *  -- Equation (E.9) with free-stream nondimensionalization applied
     * */
    real compute_strain_rate_tensor_based_dissipation_rate_from_integrated_strain_rate_tensor_magnitude_sqr (
        const real integrated_strain_rate_tensor_magnitude_sqr) const;

    /** Nondimensionalized viscous stress tensor, tau*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    template<typename real2>
    dealii::Tensor<2,dim,real2> 
    compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor (
        const real2 scaled_viscosity_coefficient,
        const dealii::Tensor<2,dim,real2> &strain_rate_tensor) const;

    /** Nondimensionalized viscous stress tensor, tau*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    template<typename real2>
    dealii::Tensor<2,dim,real2>
    compute_viscous_stress_tensor (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /// Tensor product magnitude squared
    real get_tensor_product_magnitude_sqr (
        const dealii::Tensor<2,dim,real> &tensor1,
        const dealii::Tensor<2,dim,real> &tensor2) const;

    /** Nondimensionalized Germano identity tensor, L*, from conservative solution and solution gradient
     *  Reference: Flad and Gassner 2017
     */
    dealii::Tensor<2,dim,real> compute_germano_idendity_matrix_L_component (
        const std::array<real,nstate> &conservative_soln) const;

    /** Nondimensionalized Germano identity tensor, M*, from conservative solution and solution gradient
     *  Reference: Flad and Gassner 2017
     */
    dealii::Tensor<2,dim,real> compute_germano_idendity_matrix_M_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized viscous flux (i.e. dissipative flux) dot normal vector that accounts for gradient boundary conditions
     *  References: 
     *  (1) Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4),
     *  (2) For the boundary condition case, refer to the equation above equation 458 of the following paper:
     *      Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
     */
    std::array<real,nstate> dissipative_flux_dot_normal (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &filtered_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &filtered_solution_gradient,
        const bool on_boundary,
        const dealii::types::global_dof_index cell_index,
        const dealii::Tensor<1,dim,real> &normal,
        const int boundary_type) override;

    /** Nondimensionalized viscous flux (i.e. dissipative flux) dot normal vector that accounts for gradient boundary conditions
     *  when the on boundary flag is true
     *  References: 
     *  (1) Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4),
     *  (2) For the boundary condition case, refer to the equation above equation 458 of the following paper:
     *      Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
     */
    virtual std::array<real,nstate> dissipative_flux_dot_normal_on_adiabatic_boundary (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &filtered_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &filtered_solution_gradient,
        const dealii::types::global_dof_index cell_index,
        const dealii::Tensor<1,dim,real> &normal);

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> 
    dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const override;

    /** Gradient of the scaled nondimensionalized viscosity coefficient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14 and 4.14.17)
     */
    dealii::Tensor<1,dim,real> compute_scaled_viscosity_gradient (
        const std::array<real,nstate> &primitive_soln,
        const dealii::Tensor<1,dim,real> &temperature_gradient) const;

    /** Dissipative flux Jacobian 
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal) const;

    /** Dissipative flux Jacobian wrt gradient component
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient) const;

    /// Dissipative flux contribution to the source term
    std::array<real,nstate> dissipative_source_term (
        const dealii::Point<dim,real> &pos) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time) const override;

    /// Convective flux Jacobian computed via dfad (automatic differentiation)
    /// -- Only used for verifying the dfad procedure used in dissipative flux jacobian
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian_via_dfad (
        std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Scaled viscosity coefficient derivative wrt temperature via dfad (automatic differentiation)
    /// -- Only used for verifying the basic dfad procedure that is extended in convective_flux_directional_jacobian_via_dfad()
    real compute_scaled_viscosity_coefficient_derivative_wrt_temperature_via_dfad (
        std::array<real,nstate> &conservative_soln) const;

    /** Nondimensionalized viscous flux (i.e. dissipative flux) computed 
     *  via given velocities, viscous stress tensor, and heat flux. 
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux (
        const dealii::Tensor<1,dim,real2> &vel,
        const dealii::Tensor<2,dim,real2> &viscous_stress_tensor,
        const dealii::Tensor<1,dim,real2> &heat_flux) const;

protected:

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /// Boundary face values for viscous fluxes
    virtual void boundary_face_values_viscous_flux (
        const int boundary_type,
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        const std::array<real,nstate> &/*filtered_soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*filtered_soln_grad_int*/,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;

    /** No-slip wall boundary conditions
     *  * Given by equations 460-461 of the following paper:
     *  * * Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
     */
    void boundary_wall_viscous_flux (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Evaluate the manufactured solution boundary conditions.
    void boundary_manufactured_solution (
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;

public:
    /// For post processing purposes (update comment later)
    dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &duh,
        const std::vector<dealii::Tensor<2,dim> > &dduh,
        const dealii::Tensor<1,dim>               &normals,
        const dealii::Point<dim>                  &evaluation_points) const override;
    
    /// For post processing purposes, sets the base names (with no prefix or suffix) of the computed quantities
    std::vector<std::string> post_get_names () const override;
    
    /// For post processing purposes, sets the interpretation of each computed quantity as either scalar or vector
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const override;
    
    /// For post processing purposes (update comment later)
    dealii::UpdateFlags post_get_needed_update_flags () const override;

public:
    /// Returns the square of the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    real get_tensor_magnitude_sqr (const dealii::Tensor<2,dim,real> &tensor) const;

    /// Returns the the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    real get_tensor_magnitude (const dealii::Tensor<2,dim,real> &tensor) const;

};

/// Navier-Stokes equations with constant physical source term for the turbulent channel flow case. Derived from Navier-Stokes. 
template <int dim, int nstate, typename real>
class NavierStokes_ChannelFlowConstantSourceTerm : public NavierStokes <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    NavierStokes_ChannelFlowConstantSourceTerm( 
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const bool                                                use_constant_viscosity,
        const double                                              constant_viscosity,
        const double                                              reynolds_number_based_on_friction_velocity,
        const double                                              half_channel_height,
        const double                                              temperature_inf = 273.15,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG);

    /// Nondimensional constant source term for x-momentum
    const double x_momentum_constant_source_term;

    /// Destructor
    ~NavierStokes_ChannelFlowConstantSourceTerm() {};

    /// Physical source term for turbulent channel flow case
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const override;
};

/// Wall Model Look up table
template <typename real>
class WallModelLookUpTable
{
    /** Number of different computed quantities
     *  Corresponds to the number of items in IntegratedQuantitiesEnum
     * */
    static const int NUMBER_OF_SAMPLE_POINTS = 24;
    ///< x and y data for the look up table
    static constexpr std::array<double,NUMBER_OF_SAMPLE_POINTS> yData = 
            {{0.0, 3.0, 5.0, 8.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 125.0, 150.0,
              200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 575.0, 650.0, 725.0, 800.0, 900.0, 1000.0}};
    static constexpr std::array<double,NUMBER_OF_SAMPLE_POINTS> xData = 
            {{0.0000000000000e+00, 6.6159130344294e+00, 1.6060817028163e+01, 3.4871735764788e+01,
              4.9571228279016e+01, 1.3778400828324e+02, 2.9428383798019e+02, 4.6681694258993e+02, 
              7.7798288968624e+02, 1.1109207283094e+03, 1.4603806972090e+03, 1.8230693459482e+03, 
              2.5798975867201e+03, 3.3699659153899e+03, 4.1865253644967e+03, 5.0251156052179e+03, 
              5.8825662330447e+03, 7.6450966158508e+03, 9.0023092992007e+03, 1.0385338644756e+04, 
              1.1791191697889e+04, 1.3217498477068e+04, 1.5147782638025e+04, 1.7107366776649e+04}};
public:
    WallModelLookUpTable(); ///< Constructor

    ~WallModelLookUpTable(){}; ///< Destructor

private:
    real interpolate(const real x, const bool extrapolate ) const; ///< interpolate function

public:
    /// Returns the wall shear stress magnitude calculated from the wall model
    real get_wall_shear_stress_magnitude(
        const real wall_parallel_velocity, 
        const real distance, 
        const real viscosity_coefficient,
        const real density) const;
};

/// Navier-Stokes equations with constant physical source term for the turbulent channel flow case and wall model. Derived from NavierStokes_ChannelFlowConstantSourceTerm. 
template <int dim, int nstate, typename real>
class NavierStokes_ChannelFlowConstantSourceTerm_WallModel : public NavierStokes_ChannelFlowConstantSourceTerm <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    NavierStokes_ChannelFlowConstantSourceTerm_WallModel( 
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const bool                                                use_constant_viscosity,
        const double                                              constant_viscosity,
        const double                                              reynolds_number_based_on_friction_velocity,
        const double                                              half_channel_height,
        const double                                              distance_from_wall_for_wall_model_input_velocity,
        const double                                              temperature_inf = 273.15,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG);

    /// Destructor
    ~NavierStokes_ChannelFlowConstantSourceTerm_WallModel() {};

    /// Distance from wall for wall model input velocity
    const double distance_from_wall_for_wall_model_input_velocity;

    std::unique_ptr < WallModelLookUpTable<real> > wall_model_look_up_table;

    /** Nondimensionalized viscous flux (i.e. dissipative flux) dot normal vector that accounts for gradient boundary conditions
     *  when the on boundary flag is true -- contains the wall model
     *  References: 
     *  (1) Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4),
     *  (2) For the boundary condition case, refer to the equation above equation 458 of the following paper:
     *      Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
     */
    std::array<real,nstate> dissipative_flux_dot_normal_on_adiabatic_boundary (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &filtered_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &filtered_solution_gradient,
        const dealii::types::global_dof_index cell_index,
        const dealii::Tensor<1,dim,real> &normal) override;
};

} // Physics namespace
} // PHiLiP namespace

#endif
