#include "artificial_dissipation.h"

namespace PHiLiP
{

	//=====================================================
	//  LAPLACIAN DISSIPATION FUNCTIONS 
	// ====================================================

	template <int dim, int nstate>
	template <typename real2>
	std::array<dealii::Tensor<1,dim,real2>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux_laplacian(
	const std::array<real2,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, real2 artificial_viscosity)
	{
		Physics::ConvectionDiffusion<dim,nstate,real2> CD(
		false,
		true,
		diffusion_tensor_for_artificial_dissipation,
		Parameters::ManufacturedSolutionParam::get_default_advection_vector(),
		1.0);

		std::array<dealii::Tensor<1,dim,real2>,nstate> Flux_laplacian = CD.dissipative_flux(conservative_soln, solution_gradient);

		for(int i=0;i<nstate;i++)
		{
			for(int j=0;j<dim;j++)
			{
				Flux_laplacian[i][j]*=artificial_viscosity;
			}
		}
		return Flux_laplacian;
	}

	template <int dim, int nstate> // double
	std::array<dealii::Tensor<1,dim,double>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_laplacian<double>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate>  // FadType
	std::array<dealii::Tensor<1,dim,FadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_laplacian<FadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // RadType
	std::array<dealii::Tensor<1,dim,RadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_laplacian<RadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // FadFadType
	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_laplacian<FadFadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // RadFadType
	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_laplacian<RadFadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}



	//===========================================
	//      PHYSICAL DISSIPATION FUNCTIONS
	//===========================================

	template <int dim, int nstate>
	template <typename real2>
	std::array<dealii::Tensor<1,dim,real2>,nstate>  PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux_physical(
	const std::array<real2,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, real2 artificial_viscosity)
	{
		Physics::NavierStokes<dim,nstate,real2> NS(
		input_parameters->euler_param.ref_length,
		input_parameters->euler_param.gamma_gas,
		input_parameters->euler_param.mach_inf,
		input_parameters->euler_param.angle_of_attack,
		input_parameters->euler_param.side_slip_angle,
		3.0/4.0,
		1.0);

		std::array<dealii::Tensor<1,dim,real2>,nstate> Flux_navier_stokes  =  NS.dissipative_flux(conservative_soln, solution_gradient);

		for(int i=0;i<nstate;i++)
		{
			for(int j=0;j<dim;j++)
			{
				Flux_navier_stokes[i][j]*=artificial_viscosity;
			}
		}
		return Flux_navier_stokes;
	}


	template <int dim, int nstate> // Double
	std::array<dealii::Tensor<1,dim,double>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_physical<double>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate>  // FadType
	std::array<dealii::Tensor<1,dim,FadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_physical<FadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // RadType
	std::array<dealii::Tensor<1,dim,RadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_physical<RadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // FadFadType
	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_physical<FadFadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // RadFadType
	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_physical<RadFadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	//===========================================
	//      Enthalpy Conserving DISSIPATION FUNCTIONS
	//===========================================

	template <int dim, int nstate>
	template <typename real2>
	std::array<dealii::Tensor<1,dim,real2>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux_enthalpy_conserving_laplacian(
	const std::array<real2,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, real2 artificial_viscosity)
	{
		Physics::NavierStokes<dim,nstate,real2> NS(
		input_parameters->euler_param.ref_length,
		input_parameters->euler_param.gamma_gas,
		input_parameters->euler_param.mach_inf,
		input_parameters->euler_param.angle_of_attack,
		input_parameters->euler_param.side_slip_angle,
		3.0/4.0,
		1.0);

		std::array<dealii::Tensor<1,dim,real2>,nstate> conservative_soln_gradient = solution_gradient;
		std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = NS.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
		std::array<dealii::Tensor<1,dim,real2>,nstate> diss_flux;
	
		artificial_viscosity*= NS.max_convective_eigenvalue(conservative_soln);

		for (int i=0; i<nstate; i++)
		{
			 for (int d=0; d<dim; d++)
			{
				 if(i==nstate-1)
				{
					 diss_flux[i][d] = -artificial_viscosity*(conservative_soln_gradient[i][d] + primitive_soln_gradient[i][d]);
				}
				 else
				{
					diss_flux[i][d] = -artificial_viscosity*(conservative_soln_gradient[i][d]);
				}
			 }
		 }
		 return diss_flux;
	}


	template <int dim, int nstate> // Double
	std::array<dealii::Tensor<1,dim,double>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<double>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate>  // FadType
	std::array<dealii::Tensor<1,dim,FadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<FadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // RadType
	std::array<dealii::Tensor<1,dim,RadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<RadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // FadFadType
	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<FadFadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}

	template <int dim, int nstate> // RadFadType
	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
	const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
	{
		return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<RadFadType>(conservative_soln, solution_gradient, artificial_viscosity);
	}



	template class ArtificialDissipationBase<PHILIP_DIM,1>; template class LaplacianArtificialDissipation < PHILIP_DIM,1>; 
	template class ArtificialDissipationBase<PHILIP_DIM,2>; template class LaplacianArtificialDissipation < PHILIP_DIM,2>;
	template class ArtificialDissipationBase<PHILIP_DIM,3>; template class LaplacianArtificialDissipation < PHILIP_DIM,3>;
	template class ArtificialDissipationBase<PHILIP_DIM,4>; template class LaplacianArtificialDissipation < PHILIP_DIM,4>;
	template class ArtificialDissipationBase<PHILIP_DIM,5>; template class LaplacianArtificialDissipation < PHILIP_DIM,5>;

	template class PhysicalArtificialDissipation<PHILIP_DIM,PHILIP_DIM+2>;

	template class EnthalpyConservingArtificialDissipation < PHILIP_DIM,PHILIP_DIM+2>; 

}// PHiLiP namespace
