#ifndef __NAVIER_STOKES_POTENTIAL_SOURCE__
#define __NAVIER_STOKES_POTENTIAL_SOURCE__

#include "model.h"
#include "euler.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Implementation of potential source term. Derived from the Reynolds-Averaged Navier-Stokes (RANS) equations.
template <int dim, int nstate, typename real>
class ReynoldsAveragedNavierStokes_PotentialFlow : public ReynoldsAveragedNavierStokesBase <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
	ReynoldsAveragedNavierStokes_PotentialFlow(
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
        const double                                              turbulent_prandtl_number,
        const double                                              temperature_inf = 273.15,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG);

    /// Destructor
    ~ReynoldsAveragedNavierStokes_PotentialFlow() {};

    // variables //
    /// Number of PDEs for RANS equations
    static const int nstate_navier_stokes = dim+2;

    /// Number of PDEs for RANS turbulence model
    static const int nstate_turbulence_model = nstate-(dim+2);

    /// Turbulent Prandtl number
    const double turbulent_prandtl_number;

    /// Pointer to Navier-Stokes physics object
    std::unique_ptr< NavierStokes<dim,nstate,real> > navier_stokes_physics;

    /// Physical source term
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const override;


protected:



private:
    


};

} // Physics namespace
} // PHiLiP namespace

#endif