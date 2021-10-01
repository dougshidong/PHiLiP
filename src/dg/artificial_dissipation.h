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
virtual std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux( const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)=0;
  virtual std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)=0;
  virtual std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)=0;
  virtual std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)=0;
  virtual std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)=0;

};

 template <int dim, int nstate>
 class LaplacianArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
 {
	Physics::ConvectionDiffusion<dim,nstate,double> CD_double;
	Physics::ConvectionDiffusion<dim,nstate,FadType> CD_FadType;
	Physics::ConvectionDiffusion<dim,nstate,RadType> CD_RadType;
	Physics::ConvectionDiffusion<dim,nstate,FadFadType> CD_FadFadType;
	Physics::ConvectionDiffusion<dim,nstate,RadFadType> CD_RadFadType;
 public:
 LaplacianArtificialDissipation(dealii::Tensor<2,3,double> diffusion_tensor):
 CD_double(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
 CD_FadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
 CD_RadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
 CD_FadFadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
 CD_RadFadType(false,true,diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0)
{}

//std::array<dealii::Tensor<1,dim,real>,nstate>  calc_artificial_dissipation_flux( const std::array<real,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient, real artificial_viscosity);
 
 std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux( const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity);
 std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity);

};

  template <int dim, int nstate>
  class PhysicalArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
  {
 Physics::NavierStokes<dim,nstate,double> NS_double;
 Physics::NavierStokes<dim,nstate,FadType> NS_FadType;
 Physics::NavierStokes<dim,nstate,RadType> NS_RadType;
 Physics::NavierStokes<dim,nstate,FadFadType> NS_FadFadType;
 Physics::NavierStokes<dim,nstate,RadFadType> NS_RadFadType;

  public:
  PhysicalArtificialDissipation(const Parameters::AllParameters *const parameters_input):
NS_double(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,3.0/4.0,1.0),
NS_FadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,3.0/4.0,1.0), 
NS_RadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,3.0/4.0,1.0), 
NS_FadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,3.0/4.0,1.0), 
NS_RadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,3.0/4.0,1.0) 
  {}

//std::array<dealii::Tensor<1,dim,real>,nstate>  calc_artificial_dissipation_flux( const std::array<real,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient, real artificial_viscosity);


std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux( const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity);
 std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity);

 };

  template <int dim, int nstate>
  class EnthalpyConservingArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
  {
 Physics::NavierStokes<dim,nstate,double> NS_double;
 Physics::NavierStokes<dim,nstate,FadType> NS_FadType;
 Physics::NavierStokes<dim,nstate,RadType> NS_RadType;
 Physics::NavierStokes<dim,nstate,FadFadType> NS_FadFadType;
 Physics::NavierStokes<dim,nstate,RadFadType> NS_RadFadType;

  public:
  EnthalpyConservingArtificialDissipation(const Parameters::AllParameters *const parameters_input):
NS_double(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
NS_FadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0), 
NS_RadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0), 
NS_FadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0), 
NS_RadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0) 
  {}

//std::array<dealii::Tensor<1,dim,real>,nstate>  calc_artificial_dissipation_flux( const std::array<real,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient, real artificial_viscosity);


std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux( const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity);
 std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity);
 std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux( const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity);

 };

}

#endif
