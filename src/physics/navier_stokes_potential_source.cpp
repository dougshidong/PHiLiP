#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "model.h"
#include "navier_stokes_potential_source.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Potential Flow Addition to the Navier Stokes (RANS) model
//================================================================
template <int dim, int nstate, typename real>
ReynoldsAveragedNavierStokes_PotentialFlow<dim, nstate, real>::ReynoldsAveragedNavierStokes_PotentialFlow(
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
    const double                                              temperature_inf,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : ReynoldsAveragedNavierStokesBase<dim,nstate,real>(parameters_input,
                                                        ref_length,
                                                        gamma_gas,
                                                        mach_inf,
                                                        angle_of_attack,
                                                        side_slip_angle,
                                                        prandtl_number,
                                                        reynolds_number_inf,
                                                        use_constant_viscosity,
                                                        constant_viscosity,
                                                        turbulent_prandtl_number,
                                                        temperature_inf,
                                                        isothermal_wall_temperature,
                                                        thermal_boundary_condition_type,
                                                        manufactured_solution_function,
                                                        two_point_num_flux_type)
{
    static_assert(nstate>=dim+3, "ModelBase::ReynoldsAveragedNavierStokes_PotentialFlow() should be created with nstate>=dim+3");
    // nothing to do here 
    /// (initialize diffusion / physical source here ??)
}



//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokes_PotentialFlow<dim,nstate,real>
::physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const
{
    std::array<real,nstate> physical_source;
    physical_source = this->compute_production_dissipation_cross_term(pos, conservative_soln, solution_gradient);

    /// potential source term:


    ///
    return physical_source;
}
//----------------------------------------------------------------