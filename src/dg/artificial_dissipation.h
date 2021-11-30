#ifndef _ARTIFICIALDISSIPATION_
#define _ARTIFICIALDISSIPATION_

#include "physics/physics.h"
#include "physics/convection_diffusion.h"
#include "physics/navier_stokes.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"
#include "ADTypes.hpp"

 namespace PHiLiP{

 template <int dim, int nstate>
 class ArtificialDissipationBase
 {
    public:
    virtual std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)=0;
    virtual std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)=0;
    virtual std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)=0;
    virtual std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)=0;
    virtual std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)=0;

 };




 template <int dim, int nstate>
 class LaplacianArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
 {
    //dealii::Tensor<2,3,double> diffusion_tensor_for_artificial_dissipation;
    Physics::ConvectionDiffusion<dim,nstate,double>     convection_diffusion_double;
	Physics::ConvectionDiffusion<dim,nstate,FadType>    convection_diffusion_FadType;
	Physics::ConvectionDiffusion<dim,nstate,RadType>    convection_diffusion_RadType;
	Physics::ConvectionDiffusion<dim,nstate,FadFadType> convection_diffusion_FadFadType;
	Physics::ConvectionDiffusion<dim,nstate,RadFadType> convection_diffusion_RadFadType;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_laplacian(
        const std::array<real2,nstate> &conservative_soln, 
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
        const real2 artificial_viscosity,
        const Physics::ConvectionDiffusion<dim,nstate,real2> &convection_diffusion);
 
    public:
    LaplacianArtificialDissipation(dealii::Tensor<2,3,double> diffusion_tensor): 
    //diffusion_tensor_for_artificial_dissipation(diffusion_tensor),
    convection_diffusion_double(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_FadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_RadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_FadFadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_RadFadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0)
    {}

 
    std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity);
    std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity);

 };

 

 template <int dim, int nstate>
 class PhysicalArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
 {

   // const Parameters::AllParameters *const input_parameters;
    
    Physics::NavierStokes<dim,nstate,double>     navier_stokes_double;
	Physics::NavierStokes<dim,nstate,FadType>    navier_stokes_FadType;
	Physics::NavierStokes<dim,nstate,RadType>    navier_stokes_RadType;
	Physics::NavierStokes<dim,nstate,FadFadType> navier_stokes_FadFadType;
	Physics::NavierStokes<dim,nstate,RadFadType> navier_stokes_RadFadType;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_physical(
        const std::array<real2,nstate> &conservative_soln, 
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
        const real2 artificial_viscosity,
        const Physics::NavierStokes<dim,nstate,real2> &navier_stokes);

    public:
    PhysicalArtificialDissipation(const Parameters::AllParameters *const parameters_input): //input_parameters(parameters_input) {}
    navier_stokes_double(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0)
    {}

    std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity);
    std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity);


 };



 template <int dim, int nstate>
 class EnthalpyConservingArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
 {

    //const Parameters::AllParameters *const input_parameters;
    Physics::NavierStokes<dim,nstate,double>     navier_stokes_double;
	Physics::NavierStokes<dim,nstate,FadType>    navier_stokes_FadType;
	Physics::NavierStokes<dim,nstate,RadType>    navier_stokes_RadType;
	Physics::NavierStokes<dim,nstate,FadFadType> navier_stokes_FadFadType;
	Physics::NavierStokes<dim,nstate,RadFadType> navier_stokes_RadFadType;
    
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_enthalpy_conserving_laplacian(
        const std::array<real2,nstate> &conservative_soln, 
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        real2 artificial_viscosity,
        const Physics::NavierStokes<dim,nstate,real2> &navier_stokes);

    public:
    EnthalpyConservingArtificialDissipation(const Parameters::AllParameters *const parameters_input): //input_parameters(parameters_input) {}
    navier_stokes_double(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0)
    {}

    std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity);
    std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux( 
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity);
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity);

 };

 } // namespace PHILIP

#endif
