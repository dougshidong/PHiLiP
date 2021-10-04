#include "artificial_dissipation.h"

namespace PHiLiP
{

 //=====================================================
 //  LAPLACIAN DISSIPATION FUNCTIONS 
 // ====================================================
template <int dim, int nstate> // double
std::array<dealii::Tensor<1,dim,double>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
{
    std::array<dealii::Tensor<1,dim,double>,nstate> Flux_laplacian = CD_double.dissipative_flux(conservative_soln, solution_gradient);

     for(int i=0;i<nstate;i++){
        for(int j=0;j<dim;j++){
            Flux_laplacian[i][j]*=artificial_viscosity;
        }
    }
    return Flux_laplacian;
}

template <int dim, int nstate>  // FadType
std::array<dealii::Tensor<1,dim,FadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
{
    std::array<dealii::Tensor<1,dim,FadType>,nstate> Flux_laplacian = CD_FadType.dissipative_flux(conservative_soln, solution_gradient);

    for(int i=0;i<nstate;i++){
        for(int j=0;j<dim;j++){
            Flux_laplacian[i][j]*=artificial_viscosity;
        }
    }
    return Flux_laplacian;
}

template <int dim, int nstate> // RadType
std::array<dealii::Tensor<1,dim,RadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
{
    std::array<dealii::Tensor<1,dim,RadType>,nstate> Flux_laplacian = CD_RadType.dissipative_flux(conservative_soln, solution_gradient);

    for(int i=0;i<nstate;i++) {
        for(int j=0;j<dim;j++) {
            Flux_laplacian[i][j]*=artificial_viscosity;
        }
    }
    return Flux_laplacian;
}

template <int dim, int nstate> // FadFadType
std::array<dealii::Tensor<1,dim,FadFadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
{
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate> Flux_laplacian = CD_FadFadType.dissipative_flux(conservative_soln, solution_gradient);

    for(int i=0;i<nstate;i++) {
        for(int j=0;j<dim;j++) {
            Flux_laplacian[i][j]*=artificial_viscosity; 
        }
    }
    return Flux_laplacian;
}

template <int dim, int nstate> // RadFadType
std::array<dealii::Tensor<1,dim,RadFadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
{
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate> Flux_laplacian = CD_RadFadType.dissipative_flux(conservative_soln, solution_gradient);

    for(int i=0;i<nstate;i++){
        for(int j=0;j<dim;j++){
            Flux_laplacian[i][j]*=artificial_viscosity;
        }
    }
    return Flux_laplacian;
}


//===========================================
//      PHYSICAL DISSIPATION FUNCTIONS
//===========================================

template <int dim, int nstate> // Double
std::array<dealii::Tensor<1,dim,double>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
{

	std::array<dealii::Tensor<1,dim,double>,nstate> Flux_navier_stokes  =  NS_double.dissipative_flux(conservative_soln, solution_gradient);

	for(int i=0;i<nstate;i++)
		for(int j=0;j<dim;j++)
			Flux_navier_stokes[i][j]*=artificial_viscosity;//scaled_NS_viscosity;

	return Flux_navier_stokes;

}

template <int dim, int nstate>  // FadType
std::array<dealii::Tensor<1,dim,FadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
{

	std::array<dealii::Tensor<1,dim,FadType>,nstate> Flux_navier_stokes  =  NS_FadType.dissipative_flux(conservative_soln, solution_gradient);

	for(int i=0;i<nstate;i++)
		for(int j=0;j<dim;j++)
			Flux_navier_stokes[i][j]*=artificial_viscosity;//scaled_NS_viscosity;

return Flux_navier_stokes;


}

template <int dim, int nstate> // RadType
std::array<dealii::Tensor<1,dim,RadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
{

	std::array<dealii::Tensor<1,dim,RadType>,nstate> Flux_navier_stokes  =  NS_RadType.dissipative_flux(conservative_soln, solution_gradient);

	for(int i=0;i<nstate;i++)
		for(int j=0;j<dim;j++)
			Flux_navier_stokes[i][j]*=artificial_viscosity;//scaled_NS_viscosity;

return Flux_navier_stokes;

}

template <int dim, int nstate> // FadFadType
std::array<dealii::Tensor<1,dim,FadFadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
{

	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> Flux_navier_stokes  =  NS_FadFadType.dissipative_flux(conservative_soln, solution_gradient);
	
	for(int i=0;i<nstate;i++)
		for(int j=0;j<dim;j++)
			Flux_navier_stokes[i][j]*=artificial_viscosity;//scaled_NS_viscosity;

return Flux_navier_stokes;

}

template <int dim, int nstate> // RadFadType
std::array<dealii::Tensor<1,dim,RadFadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
{

	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> Flux_navier_stokes  =  NS_RadFadType.dissipative_flux(conservative_soln, solution_gradient);
	
	for(int i=0;i<nstate;i++)
		for(int j=0;j<dim;j++)
			Flux_navier_stokes[i][j]*=artificial_viscosity;//scaled_NS_viscosity;

return Flux_navier_stokes;

}

//===========================================
//      Enthalpy Conserving DISSIPATION FUNCTIONS
//===========================================

template <int dim, int nstate> // Double
std::array<dealii::Tensor<1,dim,double>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
{
	std::array<dealii::Tensor<1,dim,double>,nstate> conservative_soln_gradient = solution_gradient;
	std::array<dealii::Tensor<1,dim,double>,nstate> primitive_soln_gradient = NS_double.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
	std::array<dealii::Tensor<1,dim,double>,nstate> diss_flux;
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

template <int dim, int nstate>  // FadType
std::array<dealii::Tensor<1,dim,FadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
{

	std::array<dealii::Tensor<1,dim,FadType>,nstate> conservative_soln_gradient = solution_gradient;
	std::array<dealii::Tensor<1,dim,FadType>,nstate> primitive_soln_gradient = NS_FadType.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
	std::array<dealii::Tensor<1,dim,FadType>,nstate> diss_flux;
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

template <int dim, int nstate> // RadType
std::array<dealii::Tensor<1,dim,RadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
{
	std::array<dealii::Tensor<1,dim,RadType>,nstate> conservative_soln_gradient = solution_gradient;
	std::array<dealii::Tensor<1,dim,RadType>,nstate> primitive_soln_gradient = NS_RadType.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
	std::array<dealii::Tensor<1,dim,RadType>,nstate> diss_flux;
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

template <int dim, int nstate> // FadFadType
std::array<dealii::Tensor<1,dim,FadFadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
{
	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> conservative_soln_gradient = solution_gradient;
	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> primitive_soln_gradient = NS_FadFadType.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
	std::array<dealii::Tensor<1,dim,FadFadType>,nstate> diss_flux;
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

template <int dim, int nstate> // RadFadType
std::array<dealii::Tensor<1,dim,RadFadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
{
	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> conservative_soln_gradient = solution_gradient;
	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> primitive_soln_gradient = NS_RadFadType.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
	std::array<dealii::Tensor<1,dim,RadFadType>,nstate> diss_flux;
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



template class ArtificialDissipationBase<PHILIP_DIM,1>; template class LaplacianArtificialDissipation < PHILIP_DIM,1>; 
template class ArtificialDissipationBase<PHILIP_DIM,2>; template class LaplacianArtificialDissipation < PHILIP_DIM,2>;
template class ArtificialDissipationBase<PHILIP_DIM,3>; template class LaplacianArtificialDissipation < PHILIP_DIM,3>;
template class ArtificialDissipationBase<PHILIP_DIM,4>; template class LaplacianArtificialDissipation < PHILIP_DIM,4>;
template class ArtificialDissipationBase<PHILIP_DIM,5>; template class LaplacianArtificialDissipation < PHILIP_DIM,5>;

template class PhysicalArtificialDissipation<PHILIP_DIM,PHILIP_DIM+2>;

template class EnthalpyConservingArtificialDissipation < PHILIP_DIM,PHILIP_DIM+2>; 

}// PHiLiP namespace
