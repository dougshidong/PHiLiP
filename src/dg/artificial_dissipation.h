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
	dealii::Tensor<2,3,double> diffusion_tensor_for_artificial_dissipation;

	template <typename real2>
	std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_laplacian(
	const std::array<real2,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, real2 artificial_viscosity);
 
	public:
	LaplacianArtificialDissipation(dealii::Tensor<2,3,double> diffusion_tensor): diffusion_tensor_for_artificial_dissipation(diffusion_tensor){}

 
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

	const Parameters::AllParameters *const input_parameters;
	template <typename real2>
	std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_physical(
	const std::array<real2,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, real2 artificial_viscosity);

	public:
	PhysicalArtificialDissipation(const Parameters::AllParameters *const parameters_input): input_parameters(parameters_input) {}

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

	const Parameters::AllParameters *const input_parameters;

	template <typename real2>
	std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_enthalpy_conserving_laplacian(
	const std::array<real2,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, real2 artificial_viscosity);

	public:
	EnthalpyConservingArtificialDissipation(const Parameters::AllParameters *const parameters_input): input_parameters(parameters_input) {}

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
