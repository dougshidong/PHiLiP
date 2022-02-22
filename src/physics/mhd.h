#ifndef __MHD__
#define __MHD__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Magnetohydrodynamics (MHD) equations. Derived from PhysicsBase
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
class MHD : public PhysicsBase <dim, nstate, real>
{
public:
    /// Constructor
    MHD(
        const double                                              gamma_gas, 
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr)
    : PhysicsBase<dim,nstate,real>(input_diffusion_tensor, manufactured_solution_function)
    , gam(gamma_gas)
    , gamm1(gam-1.0)
    {
        static_assert(nstate==8, "Physics::MHD() should be created with nstate=8");

    };
    /// Destructor
    ~MHD ()
    {};

    /// Constant heat capacity ratio of air
    const double gam;
    /// Gamma-1.0 used often
    const double gamm1;

   // double mach_inf_sqr = 1;

    //std::array<real,nstate> manufactured_solution (const dealii::Point<dim,double> &pos) const;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Convective flux: \f$ \mathbf{F}_{conv} \hat{n} \f$
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
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,//) const;
        const real /*current_time*/) const;

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

    /// Evaluate Magnetic Energy
    real compute_magnetic_energy (const std::array<real,nstate> &conservative_soln) const;

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
    real compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const;

    /// Evaluate entropy from conservative variables
    /** Note that it is not the actual entropy since it's missing some constants.
     *  Used to check entropy convergence
     *  See discussion in
     *  https://physics.stackexchange.com/questions/116779/entropy-is-constant-how-to-express-this-equation-in-terms-of-pressure-and-densi?answertab=votes#tab-top
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

    /// Convective surface split flux
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_surface_numerical_split_flux (
                const std::array< dealii::Tensor<1,dim,real>, nstate > &surface_flux,
                const std::array< dealii::Tensor<1,dim,real>, nstate > &flux_interp_to_surface) const;

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

//    void boundary_face_values (
//        const int /*boundary_type*/,
//        const dealii::Point<dim, real> &/*pos*/,
//        const dealii::Tensor<1,dim,real> &/*normal*/,
//        const std::array<real,nstate> &/*soln_int*/,
//        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
//        std::array<real,nstate> &/*soln_bc*/,
//        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
//
//    virtual dealii::Vector<double> post_compute_derived_quantities_vector (
//        const dealii::Vector<double>      &uh,
//        const std::vector<dealii::Tensor<1,dim> > &duh,
//        const std::vector<dealii::Tensor<2,dim> > &dduh,
//        const dealii::Tensor<1,dim>                  &normals,
//        const dealii::Point<dim>                  &evaluation_points) const;
//    virtual std::vector<std::string> post_get_names () const;
//    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;
//    virtual dealii::UpdateFlags post_get_needed_update_flags () const;
protected:


};

} // Physics namespace
} // PHiLiP namespace

#endif

