#ifndef __POTENTIAL_SOURCE__
#define __POTENTIAL_SOURCE__

#include "model.h"
#include "euler.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Implementation of potential source term. 
template <int dim, int nstate, typename real>
class PotentialFlowBase : public ModelBase <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    using PS_geometry_enum = Parameters::PotentialSourceParam::PotentialSourceGeometry;

    /// Constructor
	PotentialFlowBase(
        const Parameters::AllParameters *const                    parameters_input,
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
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG);

    /// Destructor
    ~PotentialFlowBase() {};

    /// Pointer to all input parameters
    const Parameters::AllParameters *const all_parameters;
    
    /// Potential Source Geometry
    const PS_geometry_enum potential_source_geometry;

    /// Variables
    const double ref_length; ///< Reference length.

    const double gamma_gas; ///< Constant heat capacity ratio of fluid.

    const double density_inf; ///< Farfield Density.

    const double angle_of_attack; ///< Angle of attack.

    /// Farfield (free stream) Reynolds number
    const double reynolds_number_inf;
    
    /// Nondimensionalized constant viscosity
    const double const_viscosity; ///< Constant viscosity of fluid

    /// Angle of trailing edge serration flaps
    const double TES_flap_angle;
    /// Lift force vector, assumes that the lift is the force in the positive y-direction.
    const dealii::Tensor<1,dim,double> lift_vector;
    /// Drag force vector, assumes that the drag is the force in the positive x-direction.
    const dealii::Tensor<1,dim,double> drag_vector;

    //// Overwriting virtual class functions ////

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Convective eigenvalues of the additional models' PDEs
    /** For potential source model, all entries are assigned to be zero */
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const override;

    /// Maximum convective eigenvalue of the additional models' PDEs
    /** For potential source model, this value is assigned to be zero */
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Maximum convective normal eigenvalue (used in Lax-Friedrichs) of the additional models' PDEs
    /** For potential source model, this value is assigned to be zero */
    real max_convective_normal_eigenvalue (
        const std::array<real,nstate> &soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;


    // /// Physical source term
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const override;

protected:
    /// zero array and tensor array
    std::array<real, nstate> zero_array; ///< Array of zeros
    std::array<dealii::Tensor<1,dim,real>,nstate> zero_tensor_array; ///< Tensor array of zeros

    // returns the area and volume of a (COMPLETE) serration ### WARNING: error introduced if not full ###
    template<typename real2>
    std::tuple<real2, real2> TES_geometry () const;

    // returns the freestream speed using the constant viscosity, density and reynolds number
    double freestream_speed () const;

    // returns lift direction unit vector
    dealii::Tensor<1,dim,double> initialize_lift_vector () const;

    // returns drag direction unit vector
    dealii::Tensor<1,dim,double> initialize_drag_vector () const;

    // computes the force acting at a point within a physical body
    dealii::Tensor<1,dim,double> compute_body_force (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

private:

};


} // Physics namespace
} // PHiLiP namespace

#endif