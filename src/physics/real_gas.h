#ifndef __REALGAS__
#define __REALGAS__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"
#include "real_gas_file_reader_and_variables/all_real_gas_constants.h"

namespace PHiLiP {
namespace Physics {

/// RealGas equations. Derived from PhysicsBase
template <int dim, int nstate, typename real> // TO DO: TEMPLATE for nspecies -- see how the LES class has nstate_baseline_physics
class RealGas : public PhysicsBase <dim, nstate, real>
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
    RealGas ( 
        const Parameters::AllParameters *const                    parameters_input,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const bool                                                has_nonzero_diffusion = false,
        const bool                                                has_nonzero_physical_source = false);

    /// Destructor
    ~RealGas() {};

    const double gam_ref; ///< reference gamma
    const double mach_ref; ///< reference mach number (Farfield Mach number)
    const double mach_ref_sqr; ///< reference mach number (Farfield Mach number squared)
    const two_point_num_flux_enum two_point_num_flux_type; ///< Two point numerical flux type (for split form)

protected:
    const double Ru; ///< universal gas constant: [J/(mol·K)]
    const double MW_Air; ///< molar weight of Air: [kg/mol]
    const double R_Air_Dim; ///< gas constant of Air: [J/(kg·K)] 
    const double R_ref; ///< reference gas constant: [J/(kg·K)] 
    const double R_Air_NonDim; ///< gas constant of Air: [NonDimensional]
    const double temperature_ref; ///< reference temperature [K]
    const double u_ref; ///< reference velocity [m/s]
    const double u_ref_sqr; ///< reference velocity squared[m/s]^2
    const double tol; ///< tolerance for NRM (Newton-raphson Method) [m/s] 
    /// Pointer to all real gas constants object for accessing the coefficients and properties (CAP)
    std::shared_ptr< PHiLiP::RealGasConstants::AllRealGasConstants > real_gas_cap;

public:
     /// Computes the entropy variables.
    std::array<real,nstate> compute_entropy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Computes the conservative variables from the entropy variables.
    std::array<real,nstate> compute_conservative_variables_from_entropy_variables (
                const std::array<real,nstate> &entropy_var) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Maximum convective normal eigenvalue (used in Lax-Friedrichs)
    /** See the book I do like CFD, equation 3.6.18 */
    real max_convective_normal_eigenvalue (
        const std::array<real,nstate> &soln,
        const dealii::Tensor<1,dim,real> &normal) const override;


    /// Maximum viscous eigenvalue.
    real max_viscous_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Dissipative flux: 0
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;

protected:
    // /// Check positive quantity and modify it according to handle_non_physical_result()
    // /** in PhysicsBase class
    //  */
    // template<typename real2>
    // bool check_positive_quantity(real2 &quantity, const std::string qty_name) const;

    /// Boundary condition handler
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

public: /// used for initial condition (initial_condition_function.cpp)
    /// Given primitive variables [density, [velocities], pressure],
    /// returns conservative variables [density, [momentum], total energy].
    ///

    /// f_S10: Convert primitive solutions to conservative solutions // TO DO: delete new and delete the original function
    std::array<real,nstate> convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const; 

/* MAIN FUNCTIONS */
protected:
    /// f_M1: Compute mixture density from conservative_soln 
    template<typename real2>
    real2 compute_mixture_density ( const std::array<real2,nstate> &conservative_soln ) const;

    /// f_M2: Compute velocities from conservative_soln 
    dealii::Tensor<1,dim,real> compute_velocities ( const std::array<real,nstate> &conservative_soln ) const;

    /// f_M3: Compute squared velocities from conservative_soln
    real compute_velocity_squared ( const std::array<real,nstate> &conservative_soln ) const;

    /// f_M4: Compute specific kinetic energy from conservative-soln
    real compute_specific_kinetic_energy ( const std::array<real,nstate> &conservative_soln ) const;

    /// f_M5: Compute mixture specific total energy from conservative_soln
    real compute_mixture_specific_total_energy ( const std::array<real,nstate> &conservative_soln ) const;

    /// f_M6: Compute species densities from conservative_soln 
    std::array<real,nstate-dim-1> compute_species_densities ( const std::array<real,nstate> &conservative_soln ) const;

    /// f_M7: Compute mass fractions from conservative_soln 
    std::array<real,nstate-dim-1> compute_mass_fractions ( const std::array<real,nstate> &conservative_soln ) const;

    /// f_M8: Compute mixture from mass fractions and species
    real compute_mixture_from_species( const std::array<real,nstate-dim-1> &mass_fractions, const std::array<real,nstate-dim-1> &species ) const;

    /// f_M9: Compute dimensional temperature from (non-dimensional) temperature
    real compute_dimensional_temperature ( const real temperature ) const;

    /// f_M9.5: Compute Rs from Ru
    std::array<real,nstate-dim-1> compute_Rs ( const real Ru ) const;

    /// f_M10: Compute species specific Cp from temperature 
    std::array<real,nstate-dim-1> compute_species_specific_Cp ( const real temperature ) const;

    /// f_M11: Compute species specific Cv from temperature 
    std::array<real,nstate-dim-1>compute_species_specific_Cv ( const real temperature ) const;

    /// f_M12: Compute species specific enthalpy from temperature 
    std::array<real,nstate-dim-1> compute_species_specific_enthalpy ( const real temperature ) const;   

    /// f_M13: Compute species specificinternal energy from temperature 
    std::array<real,nstate-dim-1> compute_species_specific_internal_energy ( const real temperature ) const;

    /// Evaluate speed of sound from conservative variables
    real compute_sound ( const std::array<real,nstate> &conservative_soln ) const;

    /// Compute convective flux from conservative_soln
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

protected:
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

};

} // Physics namespace
} // PHiLiP namespace

#endif
