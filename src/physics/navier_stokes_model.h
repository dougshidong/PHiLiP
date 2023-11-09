#ifndef __NAVIER_STOKES_MODEL__
#define __NAVIER_STOKES_MODEL__

#include "model.h"
#include "navier_stokes.h"
#include "euler.h"

namespace PHiLiP {
namespace Physics {

/// Large Eddy Simulation equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class NavierStokesWithModelSourceTerms : public ModelBase <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    NavierStokesWithModelSourceTerms(
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const bool                                                use_constant_viscosity,
        const double                                              constant_viscosity,
        const double                                              temperature_inf,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG,
        const double                                              relaxation_coefficient);

    /// Destructor
    ~NavierStokesWithModelSourceTerms() {};

    const double relaxation_coefficient; ///< Relaxation coefficient for the channel flow source term

    /// Pointer to Navier-Stokes physics object
    std::unique_ptr< NavierStokes<dim,nstate,real> > navier_stokes_physics;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ dot normal vector
    std::array<real,nstate> dissipative_flux_dot_normal (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &filtered_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &filtered_solution_gradient,
        const bool on_boundary,
        const dealii::types::global_dof_index cell_index,
        const dealii::Tensor<1,dim,real> &normal,
        const int boundary_type) const;

    /// Convective eigenvalues of the additional models' PDEs
    /** For NS model, all entries are assigned to be zero */
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const override;

    /// Maximum convective eigenvalue of the additional models' PDEs
    /** For NS model, this value is assigned to be zero */
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Maximum convective normal eigenvalue (used in Lax-Friedrichs) of the additional models' PDEs
    /** For NS model, this value is assigned to be zero */
    real max_convective_normal_eigenvalue (
        const std::array<real,nstate> &soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;

    /// Physical source term
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const override;

protected:
    std::array<real,nstate> zero_array; ///< Array of zeros
    std::array<dealii::Tensor<1,dim,real>,nstate> zero_tensor_array; ///< Tensor array of zeros

    /// Channel flow source term
    /** Forcing function to maintain the expected bulk Reynolds number throughout the solution
     *  Reference: Equation 34 of Lodato G, Castonguay P, Jameson A. Discrete filter operators for large-eddy simulation using high-order spectral difference methods. International Journal for Numerical Methods in Fluids2013;72(2):231–258. 
     */
    std::array<real,nstate> channel_flow_source_term (
        const std::array<real,nstate> &conservative_soln) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif
