#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Large Eddy Simulation (LES) Base Class
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulationBase<dim, nstate, real>::LargeEddySimulationBase( 
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number)
    : ModelBase<dim,nstate,real>() 
    , turbulent_prandtl_number(turbulent_prandtl_number)
{
    static_assert(nstate==dim+2, "PhysicsModel::LargeEddySimulationBase() should be created with nstate=dim+2");
    
    // Create NavierStokes object; even for euler_turbulence==true, LargeEddySimulation requires a NavierStokes object
    // TO DO: No need to be a shared pointer, can simply be an object, update .h file
    // std::shared_ptr< PhysicsBase<dim,dim+2,real> >  navier_stokes_physics
    //     = create_Physics(parameters_input, PDE_enum::navier_stokes);
    NavierStokes<dim,nstate,real> navier_stokes_physics 
        = NavierStokes<dim, nstate, real>(
            ref_length,
            gamma_gas,
            mach_inf,
            angle_of_attack,
            side_slip_angle,
            prandtl_number,
            reynolds_number_inf/*,
            diffusion_tensor, 
            manufactured_solution_function*/);
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
::convective_flux (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    // No additional convective terms for Large Eddy Simulation
    for (int i=0; i<nstate; i++) {
        conv_flux[i] = 0;
    }
    return conv_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> LargeEddySimulationBase<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{   
    return dissipative_flux_templated<real>(conservative_soln,solution_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real>,nstate> LargeEddySimulationBase<dim,nstate,real>
::dissipative_flux_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{
    // TO DO: Review how I'm using the template key word here for the navier_stokes_physics
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const std::array<real2,nstate> primitive_soln = navier_stokes_physics.convert_conservative_to_primitive(conservative_soln); // from Euler
    const dealii::Tensor<1,dim,real2> vel = navier_stokes_physics.extract_velocities_from_primitive(primitive_soln); // from Euler
    const std::array<dealii::Tensor<1,dim,real2>,dim> viscous_stress_tensor = compute_SGS_stress_tensor<real2>(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real2> heat_flux = compute_SGS_heat_flux<real2>(primitive_soln, primitive_soln_gradient);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux
        = navier_stokes_physics.dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux(vel,viscous_stress_tensor,heat_flux);
    
    return viscous_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution) const
{
    // TO DO: Fill in with AD here
    /* Note: Since this is only used for the manufactured solution source term, 
             the grid spacing is fixed --> No AD wrt grid
     */
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; s++)
    {
        source_term[s] = 0.0;
    }
    return source_term;
}
//----------------------------------------------------------------
//================================================================
// Smagorinsky eddy viscosity model
//================================================================
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
    const double                                              grid_spacing)
    : LargeEddySimulationBase<dim,nstate,real>(ref_length,
                                               gamma_gas,
                                               mach_inf,
                                               angle_of_attack,
                                               side_slip_angle,
                                               prandtl_number,
                                               reynolds_number_inf,
                                               turbulent_prandtl_number)
    , model_constant(model_constant)
{
    // Nothing to do here so far
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: (1) Define filter width and update CsDelta
    //        (2) Figure out how to nondimensionalize the eddy_viscosity since strain_rate_tensor is nondimensional but filter_width is not
    //        --> Solution is to simply dimensionalize the strain_rate_tensor and do eddy_viscosity/free_stream_eddy_viscosity
    //        (3) Will also have to further compute the "scaled" eddy_viscosity wrt the free stream Reynolds number
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient 
        = navier_stokes_physics.extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor 
        = navier_stokes_physics.compute_strain_rate_tensor(vel_gradient);
    
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
dealii::Tensor<1,dim,real> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    return compute_SGS_heat_flux_templated<real>(primitive_soln,primitive_soln_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: call/create the appropriate function in NavierStokes
    //        ** will have to non-dimensionalize the coefficient or dimensionalize NS then non-dimensionalize after...
    dealii::Tensor<1,dim,real2> heat_flux_SGS;
    // this->navier_stokes_physics->template 
    return heat_flux_SGS;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    return compute_SGS_stress_tensor_templated<real>(primitive_soln,primitive_soln_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: Simplify this / reduce repition of code -- create appropriate member fxns in NavierStokes

    // Compute eddy viscosity
    const real2 eddy_viscosity = compute_eddy_viscosity<real2>(primitive_soln,primitive_soln_gradient);

    // Get velocity gradients
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient 
        = navier_stokes_physics.extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    
    // Get strain rate tensor
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor 
        = navier_stokes_physics.compute_strain_rate_tensor(vel_gradient);

    // Compute the SGS stress tensor via the eddy_viscosity and the strain rate tensor
    std::array<dealii::Tensor<1,dim,real2>,dim> SGS_stress_tensor;
    SGS_stress_tensor = navier_stokes_physics.compute_viscous_stress_tensor_via_viscosity_and_strain_rate_tensor(eddy_viscosity,strain_rate_tensor);

    return SGS_stress_tensor;
}
//----------------------------------------------------------------
//================================================================
// WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_WALE<dim, nstate, real>::LargeEddySimulation_WALE( 
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              model_constant,
    const double                                              grid_spacing)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(ref_length,
                                                       gamma_gas,
                                                       mach_inf,
                                                       angle_of_attack,
                                                       side_slip_angle,
                                                       prandtl_number,
                                                       reynolds_number_inf,
                                                       turbulent_prandtl_number,
                                                       model_constant,
                                                       grid_spacing)
{
    // Nothing to do here so far
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulation_WALE<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulation_WALE<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    // TO DO: (1) Define filter width and update CsDelta
    //        (2) Figure out how to nondimensionalize the eddy_viscosity since strain_rate_tensor is nondimensional but filter_width is not
    //        --> Solution is to simply dimensionalize the strain_rate_tensor and do eddy_viscosity/free_stream_eddy_viscosity
    //        (3) Will also have to further compute the "scaled" eddy_viscosity wrt the free stream Reynolds number
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient 
        = navier_stokes_physics.extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    const std::array<dealii::Tensor<1,dim,real2>,dim> strain_rate_tensor 
        = navier_stokes_physics.compute_strain_rate_tensor(vel_gradient);
    
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
// -- LargeEddySimulationBase
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_Smagorinsky
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_WALE
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace