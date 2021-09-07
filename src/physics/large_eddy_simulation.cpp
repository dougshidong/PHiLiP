#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
LargeEddySimulationBase<dim, nstate, real>::LargeEddySimulationBase( 
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number,
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
    : NavierStokes<dim,nstate,real>(ref_length, 
                             gamma_gas, 
                             mach_inf, 
                             angle_of_attack, 
                             side_slip_angle, 
                             prandtl_number,
                             reynolds_number_inf,
                             input_diffusion_tensor, 
                             manufactured_solution_function)
    , turbulent_prandtl_number(turbulent_prandtl_number)
{
    static_assert(nstate==dim+2, "Physics::LargeEddySimulationBase() should be created with nstate=dim+2");
    // Nothing to do here so far
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> LargeEddySimulationBase<dim,nstate,real>
::compute_heat_flux (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // Get Navier-Stokes heat flux 
    const dealii::Tensor<1,dim,real2> heat_flux_navier_stokes = this->template compute_navier_stokes_heat_flux<real2>(primitive_soln, primitive_soln_gradient);
    // Get SGS heat flux
    const dealii::Tensor<1,dim,real2> heat_flux_SGS = compute_SGS_heat_flux<real2>(primitive_soln, primitive_soln_gradient);
    
    // Add SGS heat flux to Navier-Stokes heat flux
    dealii::Tensor<1,dim,real2> heat_flux;
    for (int d=0; d<dim; ++d) {
        heat_flux[d] = heat_flux_navier_stokes[d] + heat_flux_SGS[d];
    }
    return heat_flux;
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim> LargeEddySimulationBase<dim,nstate,real>
::compute_viscous_stress_tensor (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // Get viscous stress tensor from Navier-Stokes
    const std::array<dealii::Tensor<1,dim,real2>,dim> navier_stokes_stress_tensor = this->template compute_navier_stokes_stress_tensor<real2>(primitive_soln, primitive_soln_gradient);
    // Get SGS stress tensor
    const std::array<dealii::Tensor<1,dim,real2>,dim> SGS_stress_tensor = compute_SGS_stress_tensor<real2>(primitive_soln, primitive_soln_gradient);

    // Add SGS stress to Navier-Stokes stress
    std::array<dealii::Tensor<1,dim,real2>,dim> viscous_stress_tensor;
    for (int d1=0; d1<dim; ++d1) {
        for (int d2=0; d2<dim; ++d2) {
            viscous_stress_tensor[d1][d2] = navier_stokes_stress_tensor[d1][d2] + SGS_stress_tensor[d1][d2];
        }
    }
    return viscous_stress_tensor;
}

template <int dim, int nstate, typename real>
LargeEddySimulation_Smagorinsky<dim, nstate, real>::LargeEddySimulation_Smagorinsky( 
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              model_constant,
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
    : LargeEddySimulationBase<dim,nstate,real>(ref_length, 
                                               gamma_gas, 
                                               mach_inf, 
                                               angle_of_attack, 
                                               side_slip_angle, 
                                               prandtl_number,
                                               reynolds_number_inf,
                                               turbulent_prandtl_number,
                                               input_diffusion_tensor, 
                                               manufactured_solution_function)
    , model_constant(model_constant)
{
    // Nothing to do here so far
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> LargeEddySimulationBase<dim,nstate,real>
::compute_heat_flux (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // Get Navier-Stokes heat flux 
    const dealii::Tensor<1,dim,real2> heat_flux_navier_stokes = this->template compute_navier_stokes_heat_flux<real2>(primitive_soln, primitive_soln_gradient);
    // Get SGS heat flux
    const dealii::Tensor<1,dim,real2> heat_flux_SGS = compute_SGS_heat_flux<real2>(primitive_soln, primitive_soln_gradient);
    
    // Add SGS heat flux to Navier-Stokes heat flux
    dealii::Tensor<1,dim,real2> heat_flux;
    for (int d=0; d<dim; ++d) {
        heat_flux[d] = heat_flux_navier_stokes[d] + heat_flux_SGS[d];
    }
    return heat_flux;
}

template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: (1) Define filter width and update CsDelta
    //        (2) Figure out how to nondimensionalize the eddy_viscosity since strain_rate_tensor is nondimensional but filter_width is not
    //        --> Solution is to simply dimensionalize the strain_rate_tensor and do eddy_viscosity/free_stream_eddy_viscosity
    //        (3) Will also have to further compute the "scaled" eddy_viscosity wrt the free stream Reynolds number
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient = this->template extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor = this->template compute_strain_rate_tensor<real2>(vel_gradient);
    
    // Product of the model constant (Cs) and the filter width (delta)
    const real2 CsDelta = model_constant;//*filter_width;

    // Get magnitude of strain_rate_tensor
    real2 strain_rate_tensor_magnitude; // complex initializes it as 0+0i
    if(std::is_same<real2,real>::value){
        strain_rate_tensor_magnitude = 0.0;
    }
    for (int d1=0; d1<dim; ++d1) {
        for (int d2=0; d2<dim; ++d2) {
            strain_rate_tensor_magnitude += strain_rate_tensor[d1][d2]*strain_rate_tensor[d1][d2];
        }
    }
    strain_rate_tensor_magnitude = sqrt(2.0*strain_rate_tensor_magnitude);

    // Compute the eddy viscosity
    real2 eddy_viscosity = CsDelta*CsDelta*strain_rate_tensor_magnitude;
    return eddy_viscosity;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: call/create the appropriate function in NavierStokes
    dealii::Tensor<1,dim,real2> heat_flux_SGS;
    return heat_flux_SGS;
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // Compute eddy viscosity
    const real2 eddy_viscosity = compute_eddy_viscosity<real2>(primitive_soln,primitive_soln_gradient);

    // Get strain rate tensor
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient = this->template extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor = this->template compute_strain_rate_tensor<real2>(vel_gradient);

    // Compute the SGS stress tensor via the eddy_viscosity and the strain rate tensor
    std::array<dealii::Tensor<1,dim,real2>,dim> SGS_stress_tensor; // dont forget to nondim the eddy_viscosity
    //SGS_stress_tensor = this->template compute_stress_tensor_via_viscosity_and_strain_rate_tensor<real2>(eddy_viscosity,strain_rate_tensor);
    // TO DO: Define the function above in NavierStokes
    return SGS_stress_tensor;
}

// Instantiate explicitly
template class LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace