#include "artificial_dissipation.h"

namespace PHiLiP
{

template <int dim, int nstate>
ArtificialDissipationBase<dim,nstate> :: ~ArtificialDissipationBase() {}
//=====================================================
//  LAPLACIAN DISSIPATION FUNCTIONS 
//=====================================================

template <int dim, int nstate>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux_laplacian(
    const std::array<real2,nstate> &conservative_soln, 
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
    const real2 artificial_viscosity, 
    const Physics::ConvectionDiffusion<dim,nstate,real2> &convection_diffusion)
{

    std::array<dealii::Tensor<1,dim,real2>,nstate> flux_laplacian = convection_diffusion.dissipative_flux(conservative_soln, solution_gradient);

    for(int i=0;i<nstate;i++)
    {
        for(int j=0;j<dim;j++)
        {
            flux_laplacian[i][j]*=artificial_viscosity;
        }
    }
    return flux_laplacian;
}

template <int dim, int nstate> // double
std::array<dealii::Tensor<1,dim,double>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
{
    return calc_artificial_dissipation_flux_laplacian<double>(conservative_soln, solution_gradient, artificial_viscosity, convection_diffusion_double);
}

template <int dim, int nstate>  // FadType
std::array<dealii::Tensor<1,dim,FadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_laplacian<FadType>(conservative_soln, solution_gradient, artificial_viscosity, convection_diffusion_FadType);
}

template <int dim, int nstate> // RadType
std::array<dealii::Tensor<1,dim,RadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_laplacian<RadType>(conservative_soln, solution_gradient, artificial_viscosity, convection_diffusion_RadType);
}

template <int dim, int nstate> // FadFadType
std::array<dealii::Tensor<1,dim,FadFadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_laplacian<FadFadType>(conservative_soln, solution_gradient, artificial_viscosity, convection_diffusion_FadFadType);
}

template <int dim, int nstate> // RadFadType
std::array<dealii::Tensor<1,dim,RadFadType>,nstate> LaplacianArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_laplacian<RadFadType>(conservative_soln, solution_gradient, artificial_viscosity, convection_diffusion_RadFadType);
}



//===========================================
//      PHYSICAL DISSIPATION FUNCTIONS
//===========================================

template <int dim, int nstate>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate>  PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux_physical(
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
    const real2 artificial_viscosity,
    const Physics::NavierStokes<dim,nstate,real2> &navier_stokes)
{        
    std::array<dealii::Tensor<1,dim,real2>,nstate> flux_navier_stokes  =  navier_stokes.dissipative_flux(conservative_soln, solution_gradient);            

    for(int i=0;i<nstate;i++)
    {
        for(int j=0;j<dim;j++)
        {
            flux_navier_stokes[i][j]*=artificial_viscosity;
        }
    }
    return flux_navier_stokes;
}


template <int dim, int nstate> // Double
std::array<dealii::Tensor<1,dim,double>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
{
    return calc_artificial_dissipation_flux_physical<double>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_double);
}

template <int dim, int nstate>  // FadType
std::array<dealii::Tensor<1,dim,FadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_physical<FadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_FadType);
}

template <int dim, int nstate> // RadType
std::array<dealii::Tensor<1,dim,RadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_physical<RadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_RadType);
}

template <int dim, int nstate> // FadFadType
std::array<dealii::Tensor<1,dim,FadFadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_physical<FadFadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_FadFadType);
}

template <int dim, int nstate> // RadFadType
std::array<dealii::Tensor<1,dim,RadFadType>,nstate> PhysicalArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_physical<RadFadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_RadFadType);
}

//===========================================
//      ENTHALPY CONSERVING DISSIPATION FUNCTIONS
//===========================================

template <int dim, int nstate>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux_enthalpy_conserving_laplacian(
    const std::array<real2,nstate> &conservative_soln, 
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
    real2 artificial_viscosity,
    const Physics::NavierStokes<dim,nstate,real2> &navier_stokes)
{
    std::array<dealii::Tensor<1,dim,real2>,nstate> conservative_soln_gradient = solution_gradient;
    std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = navier_stokes.convert_conservative_gradient_to_primitive_gradient(conservative_soln,conservative_soln_gradient);
    std::array<dealii::Tensor<1,dim,real2>,nstate> enthalpy_diss_flux;

    artificial_viscosity*= navier_stokes.max_convective_eigenvalue(conservative_soln);

    for (int i=0; i<nstate; i++)
    {
        for (int d=0; d<dim; d++)
        {
            if(i==nstate-1)
            {
                 enthalpy_diss_flux[i][d] = -artificial_viscosity*(conservative_soln_gradient[i][d] + primitive_soln_gradient[i][d]);
            }
            else
            {
                enthalpy_diss_flux[i][d] = -artificial_viscosity*(conservative_soln_gradient[i][d]);
            }
         }
     }
     return enthalpy_diss_flux;
}


template <int dim, int nstate> // Double
std::array<dealii::Tensor<1,dim,double>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)
{
    return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<double>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_double);
}

template <int dim, int nstate>  // FadType
std::array<dealii::Tensor<1,dim,FadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<FadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_FadType);
}

template <int dim, int nstate> // RadType
std::array<dealii::Tensor<1,dim,RadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<RadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_RadType);
}

template <int dim, int nstate> // FadFadType
std::array<dealii::Tensor<1,dim,FadFadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<FadFadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_FadFadType);
}

template <int dim, int nstate> // RadFadType
std::array<dealii::Tensor<1,dim,RadFadType>,nstate> EnthalpyConservingArtificialDissipation<dim,nstate>::calc_artificial_dissipation_flux(
const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)
{
    return calc_artificial_dissipation_flux_enthalpy_conserving_laplacian<RadFadType>(conservative_soln, solution_gradient, artificial_viscosity, navier_stokes_RadFadType);
}



template class ArtificialDissipationBase<PHILIP_DIM,1>; template class LaplacianArtificialDissipation < PHILIP_DIM,1>; 
template class ArtificialDissipationBase<PHILIP_DIM,2>; template class LaplacianArtificialDissipation < PHILIP_DIM,2>;
template class ArtificialDissipationBase<PHILIP_DIM,3>; template class LaplacianArtificialDissipation < PHILIP_DIM,3>;
template class ArtificialDissipationBase<PHILIP_DIM,4>; template class LaplacianArtificialDissipation < PHILIP_DIM,4>;
template class ArtificialDissipationBase<PHILIP_DIM,5>; template class LaplacianArtificialDissipation < PHILIP_DIM,5>;

template class PhysicalArtificialDissipation<PHILIP_DIM,PHILIP_DIM+2>;

template class EnthalpyConservingArtificialDissipation < PHILIP_DIM,PHILIP_DIM+2>; 

}// PHiLiP namespace
