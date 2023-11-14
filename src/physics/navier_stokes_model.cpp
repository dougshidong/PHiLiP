#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"
#include "navier_stokes_model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Navier Stokes with Model Source Terms Class
//================================================================
template <int dim, int nstate, typename real>
NavierStokesWithModelSourceTerms<dim, nstate, real>::NavierStokesWithModelSourceTerms(
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
    const double                                              relaxation_coefficient,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : ModelBase<dim,nstate,real>(manufactured_solution_function) 
    , relaxation_coefficient(relaxation_coefficient)
    , navier_stokes_physics(std::make_unique < NavierStokes<dim,nstate,real> > (
            ref_length,
            gamma_gas,
            mach_inf,
            angle_of_attack,
            side_slip_angle,
            prandtl_number,
            reynolds_number_inf,
            use_constant_viscosity,
            constant_viscosity,
            temperature_inf,
            isothermal_wall_temperature,
            thermal_boundary_condition_type,
            manufactured_solution_function,
            two_point_num_flux_type))
{
    static_assert(nstate==dim+2, "ModelBase::NavierStokesWithModelSourceTerms() should be created with nstate=dim+2");
    // initialize zero arrays / tensors
    for (int s=0; s<nstate; ++s) {
        zero_array[s] = 0.0;
        for (int d=0; d<dim; ++d) {
            zero_tensor_array[s][d] = 0.0;
        }
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    return this->zero_tensor_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{   
    return this->zero_tensor_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::dissipative_flux_dot_normal (
        const std::array<real,nstate> &/*solution*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
        const std::array<real,nstate> &/*filtered_solution*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*filtered_solution_gradient*/,
        const bool /*on_boundary*/,
        const dealii::types::global_dof_index /*cell_index*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const int /*boundary_type*/) const
{
    std::array<real,nstate> dissipative_flux_dot_normal;
    dissipative_flux_dot_normal.fill(0.0); // initialize

    return dissipative_flux_dot_normal;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    return this->zero_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real NavierStokesWithModelSourceTerms<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    const real max_eig = 0.0;
    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real NavierStokesWithModelSourceTerms<dim,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    const real max_eig = 0.0;
    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &/*solution*/,
        const real /*current_time*/,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    return this->zero_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::physical_source_term (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    std::array<real,nstate> physical_source;
    physical_source = this->channel_flow_source_term(conservative_soln);

    return physical_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::channel_flow_source_term (
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    std::array<real,nstate> source_term;
    std::fill(source_term.begin(), source_term.end(), 0.0);

    /** Source term for driving the flow
     *  Reference: Lodato G, Castonguay P, Jameson A. Discrete filter operators for large-eddy simulation using high-order spectral difference methods. International Journal for Numerical Methods in Fluids2013;72(2):231–258. 
     */
    if(!this->navier_stokes_physics->use_constant_viscosity){
        // TO DO: use pcout and move this to the constructor
        std::cout << "ERROR: Cannot run the turbulent channel flow with a non-constant viscosity. Aborting..." << std::endl;
        std::abort();
    }
    // x-momentum term
    const real bulk_reynolds_number = this->navier_stokes_physics->reynolds_number_inf;
    const real viscosity_coefficient = this->navier_stokes_physics->constant_viscosity;
    const real scaled_viscosity_coefficient = this->navier_stokes_physics->scale_viscosity_coefficient(viscosity_coefficient);
    const real expected_mass_flow_rate = scaled_viscosity_coefficient * bulk_reynolds_number / this->half_channel_height;
    source_term[1] = this->resultant_wall_shear_force/this->domain_volume - this->relaxation_coefficient*(this->bulk_mass_flow_rate - expected_mass_flow_rate)/this->time_step;

    // energy term
    source_term[nstate-1] = this->bulk_velocity*source_term[1];
    
    return source_term;
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- NavierStokesWithModelSourceTerms
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, double >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace
