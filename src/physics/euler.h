#ifndef __EULER__
#define __EULER__

#include <deal.II/base/tensor.h>
#include "physics.h"

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
public:
    /// Constructor
    Euler (const double ref_length, const double gamma_gas, const double mach_inf, const double angle_of_attack, const double side_slip_angle)
    : ref_length(ref_length)
    , gam(gamma_gas)
    , gamm1(gam-1.0)
    , density_inf(1.0)
    , mach_inf(mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
    , angle_of_attack(angle_of_attack)
    , side_slip_angle(side_slip_angle)
    , sound_inf(1.0/(mach_inf))
    , pressure_inf(1.0/(gam*mach_inf_sqr))
    , entropy_inf(pressure_inf*pow(density_inf,-gam))
    //, internal_energy_inf(mach_inf_sqr/(gam*(gam-1.0)))
    {
        static_assert(nstate==dim+2, "Physics::Euler() should be created with nstate=dim+2");

        temperature_inf = gam*pressure_inf/density_inf * mach_inf_sqr;

        // For now, don't allow side-slip angle
        if (std::abs(side_slip_angle) >= 1e-14) {
            std::cout << "Side slip angle = " << side_slip_angle << ". Side_slip_angle must be zero. " << std::endl;
            std::cout << "I have not figured out the side slip angles just yet." << std::endl;
            std::abort();
        }
        if(dim==1) {
            velocities_inf[0] = 1.0;
        } else if(dim==2) {
            velocities_inf[0] = cos(angle_of_attack);
            velocities_inf[1] = sin(angle_of_attack); // Maybe minus??
        } else if (dim==3) {
            velocities_inf[0] = cos(angle_of_attack)*cos(side_slip_angle);
            velocities_inf[1] = sin(angle_of_attack)*cos(side_slip_angle);
            velocities_inf[2] = sin(side_slip_angle);
        }
        assert(std::abs(velocities_inf.norm() - 1.0) < 1e-14);


    };
    /// Destructor
    ~Euler ()
    {};

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
    double temperature_inf; ///< Non-dimensionalized temperature* at infinity. Should equal 1/density*(inf)

    //const double internal_energy_inf;
    /// Non-dimensionalized Velocity vector at farfield
    /** Evaluated using mach_number, angle_of_attack, and side_slip_angle.
     */
    dealii::Tensor<1,dim,double> velocities_inf; // should be const


    // dealii::Tensor<1,dim,double> compute_velocities_inf() const;

    // std::array<real,nstate> manufactured_solution (const dealii::Point<dim,double> &pos) const;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;


    /// Convective normal flux: \f$ \mathbf{F}_{conv} \cdot \hat{n} \f$
    std::array<real,nstate> convective_normal_flux (const std::array<real,nstate> &conservative_soln, const dealii::Tensor<1,dim,real> &normal) const;

    /// Convective flux Jacobian: \f$ \frac{\partial \mathbf{F}_{conv}}{\partial w} \cdot \mathbf{n} \f$
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Dissipative flux: 0
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,double> &pos,
        const std::array<real,nstate> &conservative_soln) const;

    /// Given conservative variables [density, [momentum], total energy],
    /// returns primitive variables [density, [velocities], pressure].
    ///
    /// Opposite of convert_primitive_to_conservative
    std::array<real,nstate> convert_conservative_to_primitive ( const std::array<real,nstate> &conservative_soln ) const;

    /// Given primitive variables [density, [velocities], pressure],
    /// returns conservative variables [density, [momentum], total energy].
    ///
    /// Opposite of convert_primitive_to_conservative
    std::array<real,nstate> convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const;

    /// Evaluate pressure from conservative variables
    real compute_pressure ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate pressure from conservative variables
    real compute_pressure_from_enthalpy ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate pressure from conservative variables
    real compute_specific_enthalpy ( const std::array<real,nstate> &conservative_soln, const real pressure) const;

    /// Evaluate speed of sound from conservative variables
    real compute_sound ( const std::array<real,nstate> &conservative_soln ) const;
    /// Evaluate speed of sound from density and pressure
    real compute_sound ( const real density, const real pressure ) const;

    /// Evaluate velocities from conservative variables
    dealii::Tensor<1,dim,real> compute_velocities ( const std::array<real,nstate> &conservative_soln ) const;
    /// Given the velocity vector \f$ \mathbf{u} \f$, returns the dot-product  \f$ \mathbf{u} \cdot \mathbf{u} \f$
    real compute_velocity_squared ( const dealii::Tensor<1,dim,real> &velocities ) const;

    /// Given primitive variables, returns velocities.
    dealii::Tensor<1,dim,real> extract_velocities_from_primitive ( const std::array<real,nstate> &primitive_soln ) const;
    /// Given primitive variables, returns total energy
    /** @param[in] primitive_soln    Primitive solution (density, momentum, energy)
     *  \return                      Entropy measure
     */
    real compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const;

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

    /// Given conservative variables, returns Mach number
    real compute_mach_number ( const std::array<real,nstate> &conservative_soln ) const;

    /// Given primitive variables, returns DIMENSIONALIZED temperature using the equation of state
    real compute_dimensional_temperature ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given primitive variables, returns NON-DIMENSIONALIZED temperature using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_temperature ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given pressure and temperature, returns NON-DIMENSIONALIZED density using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_density_from_pressure_temperature ( const real pressure, const real temperature ) const;

    /// Given density and pressure, returns NON-DIMENSIONALIZED temperature using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_temperature_from_density_pressure ( const real density, const real pressure ) const;

    /// The Euler split form is that of Kennedy & Gruber.
    /** Refer to Gassner's paper (2016) Eq. 3.10 for more information:  */
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

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

    /// Mean specific energy given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_specific_energy(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, double> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

    virtual dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>      &uh,
        const std::vector<dealii::Tensor<1,dim> > &duh,
        const std::vector<dealii::Tensor<2,dim> > &dduh,
        const dealii::Tensor<1,dim>                  &normals,
        const dealii::Point<dim>                  &evaluation_points) const;
    virtual std::vector<std::string> post_get_names () const;
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;
    virtual dealii::UpdateFlags post_get_needed_update_flags () const;
protected:


};

} // Physics namespace
} // PHiLiP namespace

#endif
