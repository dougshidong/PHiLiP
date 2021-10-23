#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "physics/physics.h"
#include "physics/euler.h"
#include "physics/navier_stokes.h"

#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Large Eddy Simulation (LES) Base Class
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulationBase<dim, nstate, real>::LargeEddySimulationBase( 
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
    const double                                                     turbulent_prandtl_number)
    : ModelBase<dim,nstate,real>() 
    , navier_stokes_physics(navier_stokes_physics_input)
    , turbulent_prandtl_number(turbulent_prandtl_number)
{
    static_assert(nstate==dim+2, "PhysicsModel::LargeEddySimulationBase() should be created with nstate=dim+2");
    // Nothing to do here so far
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulationBase<dim,nstate,real>
::get_tensor_magnitude_sqr (
    const std::array<dealii::Tensor<1,dim,real2>,dim> &tensor) const
{
    real2 tensor_magnitude; // complex initializes it as 0+0i
    if(std::is_same<real2,real>::value){
        tensor_magnitude = 0.0;
    }
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            tensor_magnitude += tensor[i][j]*tensor[i][j];
        }
    }
    return tensor_magnitude;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> LargeEddySimulationBase<dim,nstate,real>
::model_convective_flux (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> model_conv_flux;
    // No additional convective terms for Large Eddy Simulation
    for (int i=0; i<nstate; i++) {
        model_conv_flux[i] = 0;
    }
    return model_conv_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>    
std::array<dealii::Tensor<1,dim,real>,nstate> model_dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    // TO DO: Review how i'm using the template key word here for the navier_stokes_physics
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<1,dim,real2> vel = this->navier_stokes_physics->template extract_velocities_from_primitive<real2>(primitive_soln); // from Euler
    const std::array<dealii::Tensor<1,dim,real2>,dim> viscous_stress_tensor = compute_SGS_stress_tensor<real2>(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real2> heat_flux = compute_SGS_heat_flux<real2>(primitive_soln, primitive_soln_gradient);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux 
        = this->navier_stokes_physics
              ->template dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux_templated<real2> (
                vel,
                viscous_stress_tensor,
                heat_flux);
    
    return viscous_flux;
}
//----------------------------------------------------------------
//================================================================
// Smagorinsky eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_Smagorinsky<dim, nstate, real>::LargeEddySimulation_Smagorinsky( 
    const double                                                     model_constant,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
    const double                                                     turbulent_prandtl_number)
    : LargeEddySimulationBase<dim,nstate,real>(navier_stokes_physics_input,
                                               turbulent_prandtl_number)
    , model_constant(model_constant)
{
    // Nothing to do here so far
}
//----------------------------------------------------------------
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
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient 
        = this->navier_stokes_physics
              ->template extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor 
        = this->navier_stokes_physics->template compute_strain_rate_tensor<real2>(vel_gradient);
    
    // Product of the model constant (Cs) and the filter width (delta)
    const real2 CsDelta = model_constant;//*filter_width;
    // Get magnitude of strain_rate_tensor
    const real2 strain_rate_tensor_magnitude_sqr = this->template get_tensor_magnitude_sqr<real2>(strain_rate_tensor);
    // Compute the eddy viscosity
    const real2 eddy_viscosity = CsDelta*CsDelta*sqrt(2.0*strain_rate_tensor_magnitude_sqr);

    return eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: call/create the appropriate function in NavierStokes
    //        ** will have to non-dimensionalize the coefficient or dimensionalize NS then non-dimensionalize after...
    dealii::Tensor<1,dim,real2> heat_flux_SGS = ;
    // this->navier_stokes_physics->template 
    return heat_flux_SGS;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: Simplify this / reduce repition of code -- create appropriate member fxns in NavierStokes

    // Compute eddy viscosity
    const real2 eddy_viscosity = compute_eddy_viscosity<real2>(primitive_soln,primitive_soln_gradient);

    // Get velocity gradients
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient 
        = this->navier_stokes_physics
              ->template extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    
    // Get strain rate tensor
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor 
        = this->navier_stokes_physics
              ->template compute_strain_rate_tensor<real2>(vel_gradient);

    // Compute the SGS stress tensor via the eddy_viscosity and the strain rate tensor
    std::array<dealii::Tensor<1,dim,real2>,dim> SGS_stress_tensor;
    SGS_stress_tensor = this->navier_stokes_physics->template compute_stress_tensor_via_viscosity_and_strain_rate_tensor<real2>(eddy_viscosity,strain_rate_tensor);

    return SGS_stress_tensor;
}
//----------------------------------------------------------------
//================================================================
// WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_WALE<dim, nstate, real>::LargeEddySimulation_WALE( 
    const double                                                     model_constant,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
    const double                                                     turbulent_prandtl_number)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(model_constant,
                                                       navier_stokes_physics_input,
                                                       turbulent_prandtl_number)
{
    // Nothing to do here so far
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulation_WALE<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: (1) Define filter width and update CsDelta
    //        (2) Figure out how to nondimensionalize the eddy_viscosity since strain_rate_tensor is nondimensional but filter_width is not
    //        --> Solution is to simply dimensionalize the strain_rate_tensor and do eddy_viscosity/free_stream_eddy_viscosity
    //        (3) Will also have to further compute the "scaled" eddy_viscosity wrt the free stream Reynolds number
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient 
        = this->navier_stokes_physics
              ->template extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor 
        = this->navier_stokes_physics->template compute_strain_rate_tensor<real2>(vel_gradient);
    
    // Product of the model constant (Cs) and the filter width (delta)
    const real2 CwDelta = model_constant;//*filter_width;
    // Get deviatoric stresss tensor
    std::array<dealii::Tensor<1,dim,real2>,dim> g_sqr; // $g_{ij}^{2}$
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            
            real2 val;if(std::is_same<real2,real>::value){val = 0.0;}

            for (int k=0; k<dim; ++k) {
                val += vel_gradient[i][k]*vel_gradient[k][j];
            }
            g_sqr[i][j] = val;
        }
    }
    real2 val;if(std::is_same<real2,real>::value){val = 0.0;}
    for (int k=0; k<dim; ++k) {
        trace_g_sqr += g_sqr[k][k];
    }
    std::array<dealii::Tensor<1,dim,real2>,dim> deviatoric_strain_rate_tensor;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            deviatoric_strain_rate_tensor[i][j] = 0.5*(g_sqr[i][j]+g_sqr[j][i]);
        }
    }
    for (int k=0; k<dim; ++k) {
        deviatoric_strain_rate_tensor[k][k] += -(1.0/3.0)*trace_g_sqr;
    }
    
    // Get magnitude of strain_rate_tensor and deviatoric_strain_rate_tensor
    const real2 strain_rate_tensor_magnitude_sqr            = this->template get_tensor_magnitude_sqr<real2>(strain_rate_tensor);
    const real2 deviatoric_strain_rate_tensor_magnitude_sqr = this->template get_tensor_magnitude_sqr<real2>(deviatoric_strain_rate_tensor);
    // Compute the eddy viscosity
    const real2 eddy_viscosity = CwDelta*CwDelta*pow(deviatoric_strain_rate_tensor_magnitude_sqr,1.5)/(pow(strain_rate_tensor_magnitude_sqr,2.5) + pow(deviatoric_strain_rate_tensor_magnitude_sqr,1.25));

    return eddy_viscosity;
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
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