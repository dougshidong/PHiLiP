#ifndef __EULER__
#define __EULER__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Euler equations. Derived from PhysicsBase
/** Only 2D and 3D
 *  State variable and convective fluxes given by
 *
 *  \f[ 
 *  \mathbf{w} = 
 *  \begin{bmatrix} \rho \\ \rho v_1 \\ \rho v_2 \\ \rho v_3 \\ \rho E \end{bmatrix}
 *  , \qquad
 *  \mathbf{F}_{conv} = 
 *  \begin{bmatrix} 
 *      \mathbf{f}^x_{conv}, \mathbf{f}^y_{conv}, \mathbf{f}^z_{conv}
 *  \end{bmatrix}
 *  =
 *  \begin{bmatrix} 
 *  \begin{bmatrix} 
 *  \rho v_1 \\
 *  \rho v_1 v_1 + p \\
 *  \rho v_1 v_2     \\ 
 *  \rho v_1 v_3     \\
 *  v_1 (\rho e+p)
 *  \end{bmatrix}
 *  ,
 *  \begin{bmatrix} 
 *  \rho v_2 \\
 *  \rho v_1 v_2     \\
 *  \rho v_2 v_2 + p \\ 
 *  \rho v_2 v_3     \\
 *  v_2 (\rho e+p)
 *  \end{bmatrix}
 *  ,
 *  \begin{bmatrix} 
 *  \rho v_3 \\
 *  \rho v_1 v_3     \\
 *  \rho v_2 v_3     \\ 
 *  \rho v_3 v_3 + p \\
 *  v_3 (\rho e+p)
 *  \end{bmatrix}
 *  \end{bmatrix} \f]
 *  
 *  where, \f$ E \f$ is the specific total energy and \f$ e \f$ is the specific internal
 *  energy, related by
 *  \f[
 *      E = e + |V|^2 / 2
 *  \f] 
 *  For a calorically perfect gas
 *
 *  \f[
 *  p=(\gamma -1)(\rho e-\frac{1}{2}\rho \|\mathbf{v}\|)
 *  \f]
 *
 *  Dissipative flux \f$ \mathbf{F}_{diss} = \mathbf{0} \f$
 *
 *  Source term \f$ s(\mathbf{x}) \f$
 *
 *  Equation:
 *  \f[ \boldsymbol{\nabla} \cdot
 *         (  \mathbf{F}_{conv}( w ) 
 *          + \mathbf{F}_{diss}( w, \boldsymbol{\nabla}(w) )
 *      = s(\mathbf{x})
 *  \f]
 *
 *
 *  Still need to provide functions to un-non-dimensionalize the variables.
 *  Like, given density_inf
 */
template <int dim, int nstate, typename real>
class Euler : public PhysicsBase <dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following: */
    using PhysicsBase<dim,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nstate,real>::source_term;
public:
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    Euler ( 
        const Parameters::AllParameters *const                    parameters_input,
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG,
        const bool                                                has_nonzero_diffusion = false,
        const bool                                                has_nonzero_physical_source = false);

    const double ref_length; ///< Reference length.
    const double gam; ///< Constant heat capacity ratio of fluid.
    const double gamm1; ///< Constant heat capacity ratio (Gamma-1.0) used often.

    /// Non-dimensionalized density* at infinity. density* = density/density_ref
    /// Choose density_ref = density(inf)
    /// density*(inf) = density(inf) / density_ref = density(inf)/density(inf) = 1.0
    const double density_inf;

    const double mach_inf; ///< Farfield Mach number.
    const double mach_inf_sqr; ///< Farfield Mach number squared.
    /// Angle of attack.
    /** Mandatory for 2D simulations.
     */
    const double angle_of_attack;
    /// Sideslip angle.
    /** Mandatory for 2D and 3D simulations.
     */
    const double side_slip_angle;


    const double sound_inf; ///< Non-dimensionalized sound* at infinity
    const double pressure_inf; ///< Non-dimensionalized pressure* at infinity
    const double entropy_inf; ///< Entropy measure at infinity
    const two_point_num_flux_enum two_point_num_flux_type; ///< Two point numerical flux type (for split form)
    double temperature_inf; ///< Non-dimensionalized temperature* at infinity. Should equal 1/density*(inf)
    double dynamic_pressure_inf; ///< Non-dimensionalized dynamic pressure* at infinity

    //const double internal_energy_inf;
    /// Non-dimensionalized Velocity vector at farfield
    /** Evaluated using mach_number, angle_of_attack, and side_slip_angle.
     */
    dealii::Tensor<1,dim,double> velocities_inf; // should be const


    // dealii::Tensor<1,dim,double> compute_velocities_inf() const;

    // std::array<real,nstate> manufactured_solution (const dealii::Point<dim,double> &pos) const;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const override;

    /// Convective normal flux: \f$ \mathbf{F}_{conv} \cdot \hat{n} \f$
    std::array<real,nstate> convective_normal_flux (const std::array<real,nstate> &conservative_soln, const dealii::Tensor<1,dim,real> &normal) const;

    /// Convective flux Jacobian: \f$ \frac{\partial \mathbf{F}_{conv}}{\partial w} \cdot \mathbf{n} \f$
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const override;

    /// Maximum convective eigenvalue
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const override;

    /// Maximum convective normal eigenvalue (used in Lax-Friedrichs)
    /** See the book I do like CFD, equation 3.6.18 */
    real max_convective_normal_eigenvalue (
        const std::array<real,nstate> &soln,
        const dealii::Tensor<1,dim,real> &normal) const override;

    /// Maximum viscous eigenvalue.
    real max_viscous_eigenvalue (const std::array<real,nstate> &soln) const override;

    /// Dissipative flux: 0
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// (function overload) Dissipative flux: 0
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;

    /// (function overload) Source term is zero or depends on manufactured solution
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time) const;

    /// Convective flux contribution to the source term
    std::array<real,nstate> convective_source_term (
        const dealii::Point<dim,real> &pos) const;

protected:
    /// Check positive quantity and modify it according to handle_non_physical_result()
    /** in PhysicsBase class
     */
    template<typename real2>
    bool check_positive_quantity(real2 &quantity, const std::string qty_name) const;

public:
    /// Given conservative variables [density, [momentum], total energy],
    /// returns primitive variables [density, [velocities], pressure].
    ///
    /// Opposite of convert_primitive_to_conservative
    template<typename real2>
    std::array<real2,nstate> convert_conservative_to_primitive ( const std::array<real2,nstate> &conservative_soln ) const;

    /// Given primitive variables [density, [velocities], pressure],
    /// returns conservative variables [density, [momentum], total energy].
    ///
    /// Opposite of convert_primitive_to_conservative
    std::array<real,nstate> convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const;

    /// Evaluate pressure from conservative variables
    template<typename real2>
    real2 compute_pressure ( const std::array<real2,nstate> &conservative_soln ) const;

    /// Evaluate physical entropy = log(p \rho^{-\gamma}) from pressure and density
    template<typename real2>
    real2 compute_entropy (const real2 density, const real2 pressure) const;

    /// Evaluate pressure from conservative variables
    real compute_specific_enthalpy ( const std::array<real,nstate> &conservative_soln, const real pressure) const;

    /// Compute numerical entropy function -rho s 
    real compute_numerical_entropy_function(const std::array<real,nstate> &conservative_soln) const;

    /// Evaluate speed of sound from conservative variables
    real compute_sound ( const std::array<real,nstate> &conservative_soln ) const;
    /// Evaluate speed of sound from density and pressure
    real compute_sound ( const real density, const real pressure ) const;

    /// Evaluate velocities from conservative variables
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_velocities ( const std::array<real2,nstate> &conservative_soln ) const;
    /// Given the velocity vector \f$ \mathbf{u} \f$, returns the dot-product  \f$ \mathbf{u} \cdot \mathbf{u} \f$
    template<typename real2>
    real2 compute_velocity_squared ( const dealii::Tensor<1,dim,real2> &velocities ) const;

    /// Given primitive variables, returns velocities.
    template<typename real2>
    dealii::Tensor<1,dim,real2> extract_velocities_from_primitive ( const std::array<real2,nstate> &primitive_soln ) const;
    /// Given primitive variables, returns total energy
    /** @param[in] primitive_soln    Primitive solution (density, momentum, energy)
     *  \return                      Entropy measure
     */
    real compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given primitive variables, returns kinetic energy
    real compute_kinetic_energy_from_primitive_solution ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given conservative variables, returns kinetic energy
    real compute_kinetic_energy_from_conservative_solution ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate entropy from conservative variables
    /** Note that it is not the actual entropy since it's missing some constants.
     *  Used to check entropy convergence
     *  See discussion in
     *  https://physics.stackexchange.com/questions/116779/entropy-is-constant-how-to-express-this-equation-in-terms-of-pressure-and-densi?answertab=votes#tab-top
     *
     *  @param[in] conservative_soln Conservative solution (density, momentum, energy)
     *  \return                      Entropy measure
     */
    real compute_entropy_measure ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate entropy from density and pressure. 
    real compute_entropy_measure ( const real density, const real pressure ) const;

    /// Given conservative variables, returns Mach number
    real compute_mach_number ( const std::array<real,nstate> &conservative_soln ) const;

    /// Given primitive variables, returns NON-DIMENSIONALIZED temperature using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    template<typename real2>
    real2 compute_temperature ( const std::array<real2,nstate> &primitive_soln ) const;

    /// Given pressure and temperature, returns NON-DIMENSIONALIZED density using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_density_from_pressure_temperature ( const real pressure, const real temperature ) const;

    /// Given density and pressure, returns NON-DIMENSIONALIZED temperature using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_temperature_from_density_pressure ( const real density, const real pressure ) const;

    /// Given density and temperature, returns NON-DIMENSIONALIZED pressure using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_pressure_from_density_temperature ( const real density, const real temperature ) const;

    ///  Evaluates convective flux based on the chosen split form.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const override;

    /// Computes the entropy variables.
    /// Given conservative variables [density, [momentum], total energy],
    /// Computes entropy variables according to Chan 2018, eq. 119
    std::array<real,nstate> compute_entropy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Computes the conservative variables [density, [momentum], total energy
    /// from the entropy variables according to Chan 2018, eq. 120
    std::array<real,nstate> compute_conservative_variables_from_entropy_variables (
                const std::array<real,nstate> &entropy_var) const;

    /// Computes the kinetic energy variables.
    std::array<real,nstate> compute_kinetic_energy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Mean density given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_density(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Mean pressure given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_pressure(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Mean velocities given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    dealii::Tensor<1,dim,real> compute_mean_velocities(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Mean specific total energy given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_specific_total_energy(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Boundary condition handler
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

    /// For post processing purposes (update comment later)
    virtual dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &duh,
        const std::vector<dealii::Tensor<2,dim> > &dduh,
        const dealii::Tensor<1,dim>               &normals,
        const dealii::Point<dim>                  &evaluation_points) const;
    
    /// For post processing purposes, sets the base names (with no prefix or suffix) of the computed quantities
    virtual std::vector<std::string> post_get_names () const;
    
    /// For post processing purposes, sets the interpretation of each computed quantity as either scalar or vector
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;
    
    /// For post processing purposes (update comment later)
    virtual dealii::UpdateFlags post_get_needed_update_flags () const;

protected:
    /** Slip wall boundary conditions (No penetration)
     *  * Given by Algorithm II of the following paper:
     *  * * Krivodonova, L., and Berger, M.,
     *      “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
     *      Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
     */
    void boundary_slip_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Wall boundary condition
    virtual void boundary_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Evaluate the manufactured solution boundary conditions.
    virtual void boundary_manufactured_solution (
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Pressure Outflow Boundary Condition (back pressure)
    /// Reference: Carlson 2011, sec. 2.4
    void boundary_pressure_outflow (
        const real total_inlet_pressure,
        const real back_pressure,
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc) const;

    /// Inflow boundary conditions (both subsonic and supersonic)
    /// Reference: Carlson 2011, sec. 2.2 & sec 2.9
    void boundary_inflow (
        const real total_inlet_pressure,
        const real total_inlet_temperature,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc) const;

    /// Riemann-based farfield boundary conditions based on freestream values.
    /// Reference: ? (ask Doug)
    void boundary_riemann (
       const dealii::Tensor<1,dim,real> &normal_int,
       const std::array<real,nstate> &soln_int,
       std::array<real,nstate> &soln_bc) const;

    /// Simple farfield boundary conditions based on freestream values
    void boundary_farfield (
        std::array<real,nstate> &soln_bc) const;

    /// Wall boundary condition
    void boundary_do_nothing (
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Custom boundary conditions for the left boundary of the astrophysical mach jet case where it is not hypersonic inflow.
    void boundary_custom (
        std::array<real,nstate> &soln_bc) const;

    /// Boundary conditions based on user-defined values
    void boundary_astrophysical_inflow (
        std::array<real,nstate> &soln_bc) const;

    /// Get manufactured solution value
    std::array<real,nstate> get_manufactured_solution_value(
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution gradient
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient(
        const dealii::Point<dim,real> &pos) const;

    /** Entropy conserving split form flux of Kennedy and Gruber.
     *  Refer to Gassner's paper (2016) Eq. 3.10  */
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_kennedy_gruber (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Compute Ismail-Roe parameter vector from primitive solution
    std::array<real,nstate> compute_ismail_roe_parameter_vector_from_primitive(
        const std::array<real,nstate> &primitive_soln) const;

    /// Compute Ismail-Roe logarithmic mean
    real compute_ismail_roe_logarithmic_mean(const real val1, const real val2) const;

    /** Entropy conserving split form flux of Ismail & Roe.
     *  Refer to Gassner's paper (2016) Eq. 3.17  */
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_ismail_roe (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Chandrashekar entropy conserving flux.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_chandrashekar (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Ranocha pressure equilibrium preserving, entropy and energy conserving flux.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_ranocha (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif
