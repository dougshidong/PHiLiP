#ifndef _ARTIFICIALDISSIPATION_
#define _ARTIFICIALDISSIPATION_

#include "physics/physics.h"
#include "physics/convection_diffusion.h"
#include "physics/navier_stokes.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"
#include "ADTypes.hpp"

namespace PHiLiP{

/// Class to add artificial dissipation with an option to add one of the 3 dissipation types: 1. Laplacian, 2. Physical and 3. Enthalpy Laplacian. 

template <int dim, int nstate>
class ArtificialDissipationBase
{
    public:
    /// Identity diffusion tensor for Laplace artificial dissipation.
    dealii::Tensor<2,3,double> diffusion_tensor;

    /// Virtual fuction overloaded with type double.
    virtual std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity)=0;
    
    /// Virtual fuction overloaded with type FadType.
    virtual std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity)=0;
    
    /// Virtual fuction overloaded with type RadType.
    virtual std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity)=0;
    
    /// Virtual fuction overloaded with type FadFadType.
    virtual std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity)=0;
    
    /// Virtual fuction overloaded with type RadFadType.
    virtual std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity)=0;

    /// Constructor of ArtificialDissipationBase
    ArtificialDissipationBase()
    {
        for(int i=0; i<3; i++)
        {
            diffusion_tensor[i][i] = 1.0;
        }
    }

    /// Virtual destructor of ArtificialDissipationBase
    virtual ~ArtificialDissipationBase() = 0;

};



/// Adds Laplacian artificial dissipation (from Persson and Peraire, 2008).
template <int dim, int nstate>
class LaplacianArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
{
    /// ConvectionDiffusion object of type double.
    Physics::ConvectionDiffusion<dim,nstate,double>     convection_diffusion_double;
    
    /// ConvectionDiffusion object of type FadType.
    Physics::ConvectionDiffusion<dim,nstate,FadType> convection_diffusion_FadType;
    
    /// ConvectionDiffusion object of type RadType.
    Physics::ConvectionDiffusion<dim,nstate,RadType>    convection_diffusion_RadType;
    
    /// ConvectionDiffusion object of type FadFadType.
    Physics::ConvectionDiffusion<dim,nstate,FadFadType> convection_diffusion_FadFadType;
    
    /// ConvectionDiffusion object of type RadFadType.
    Physics::ConvectionDiffusion<dim,nstate,RadFadType> convection_diffusion_RadFadType;


    template <typename real2>
    /// Calculates laplacian flux
    std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_laplacian(
        const std::array<real2,nstate> &conservative_soln, 
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
        const real2 artificial_viscosity,
        const Physics::ConvectionDiffusion<dim,nstate,real2> &convection_diffusion);
 
    public:
    /// Constructor of LaplacianArtificialDissipation.
    LaplacianArtificialDissipation(): 
    convection_diffusion_double(false,true,this->diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_FadType(false,true,this->diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_RadType(false,true,this->diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_FadFadType(false,true,this->diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0),
    convection_diffusion_RadFadType(false,true,this->diffusion_tensor,Parameters::ManufacturedSolutionParam::get_default_advection_vector(),1.0)
    {}

    /// Destructor of LaplacianArtificialDissipation
    ~LaplacianArtificialDissipation() {};

 
    /// Laplacian flux function overloaded with type double.
    std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity) override;
    
    /// Laplacian flux function overloaded with type FadType.
    std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity) override;
    
    /// Laplacian flux function overloaded with type RadType.
    std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity) override;
    
    /// Laplacian flux function overloaded with type FadFadType.
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity) override;
    
    /// Laplacian flux function overloaded with type RadFadType.
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity) override;

};

 
/// Adds Physical artificial dissipation (from Persson and Peraire, 2008).
template <int dim, int nstate>
class PhysicalArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
{
   
    /// NavierStokes object of type double.
    Physics::NavierStokes<dim,nstate,double>     navier_stokes_double;
    
    /// NavierStokes object of type FadType.
    Physics::NavierStokes<dim,nstate,FadType>    navier_stokes_FadType;
    
    /// NavierStokes object of type RadType.
    Physics::NavierStokes<dim,nstate,RadType>    navier_stokes_RadType;
    
    /// NavierStokes object of type FadFadType.
    Physics::NavierStokes<dim,nstate,FadFadType> navier_stokes_FadFadType;
    
    /// NavierStokes object of type RadFadType.
    Physics::NavierStokes<dim,nstate,RadFadType> navier_stokes_RadFadType;

    template <typename real2>
    /// Calculates navier stokes artificial dissipation flux.
    std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_physical(
        const std::array<real2,nstate> &conservative_soln, 
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient, 
        const real2 artificial_viscosity,
        const Physics::NavierStokes<dim,nstate,real2> &navier_stokes);

    public:
    /// Constructor of PhysicalArtificialDissipation.
    PhysicalArtificialDissipation(const Parameters::AllParameters *const parameters_input): //input_parameters(parameters_input) {}
    navier_stokes_double(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0)
    {}

    /// Destructor of PhysicalArtificialDissipation
    ~PhysicalArtificialDissipation() {};

    /// Physical flux function overloaded with type double.
    std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity) override;
    
    /// Physical flux function overloaded with type FadType.
    std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity) override;
    
    /// Physical flux function overloaded with type RadType.
    std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity) override;
    
    /// Physical flux function overloaded with type FadFadType.
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity) override;
    
    /// Physical flux function overloaded with type adFadType.
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity) override;


};



/// Adds enthalpy laplacian artificial dissipation (from G.E. Barter and D.L. Darmofal, 2009).
template <int dim, int nstate>
class EnthalpyConservingArtificialDissipation: public ArtificialDissipationBase <dim, nstate>
{

    /// NavierStokes object of type double.
    Physics::NavierStokes<dim,nstate,double>     navier_stokes_double;
    
    /// NavierStokes object of type FadType.
    Physics::NavierStokes<dim,nstate,FadType>    navier_stokes_FadType;
    
    /// NavierStokes object of type RadType.
    Physics::NavierStokes<dim,nstate,RadType>    navier_stokes_RadType;
    
    /// NavierStokes object of type FadFadType.
    Physics::NavierStokes<dim,nstate,FadFadType> navier_stokes_FadFadType;
    
    /// NavierStokes object of type RadFadType.
    Physics::NavierStokes<dim,nstate,RadFadType> navier_stokes_RadFadType;
    
    template <typename real2>
    /// Calculates enthalpy laplacian artificial dissipation flux.
    std::array<dealii::Tensor<1,dim,real2>,nstate>  calc_artificial_dissipation_flux_enthalpy_conserving_laplacian(
        const std::array<real2,nstate> &conservative_soln, 
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        real2 artificial_viscosity,
        const Physics::NavierStokes<dim,nstate,real2> &navier_stokes);

    public:
    /// Constructor of EnthalpyConservingArtificialDissipation
    EnthalpyConservingArtificialDissipation(const Parameters::AllParameters *const parameters_input): //input_parameters(parameters_input) {}
    navier_stokes_double(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_FadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0),
    navier_stokes_RadFadType(parameters_input->euler_param.ref_length,parameters_input->euler_param.gamma_gas,parameters_input->euler_param.mach_inf,parameters_input->euler_param.angle_of_attack, parameters_input->euler_param.side_slip_angle,0.75,1.0)
    {}

    /// Destructor of EnthalpyConservingArtificialDissipation
    ~EnthalpyConservingArtificialDissipation() {};

    /// Enthalpy laplacian flux function overloaded with type double.
    std::array<dealii::Tensor<1,dim,double>,nstate>  calc_artificial_dissipation_flux(
    const std::array<double,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient, double artificial_viscosity) override;
    
    /// Enthalpy laplacian flux function overloaded with type FadType.
    std::array<dealii::Tensor<1,dim,FadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient, FadType artificial_viscosity) override;
    
    /// Enthalpy laplacian flux function overloaded with type RadType.
    std::array<dealii::Tensor<1,dim,RadType>,nstate>  calc_artificial_dissipation_flux( 
    const std::array<RadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadType>,nstate> &solution_gradient, RadType artificial_viscosity) override;
    
    /// Enthalpy laplacian flux function overloaded with type FadFadType.
    std::array<dealii::Tensor<1,dim,FadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<FadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient, FadFadType artificial_viscosity) override;
    
    /// Enthalpy laplacian flux function overloaded with type RadFadType.
    std::array<dealii::Tensor<1,dim,RadFadType>,nstate>  calc_artificial_dissipation_flux(
    const std::array<RadFadType,nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient, RadFadType artificial_viscosity) override;

};
} // namespace PHILIP

#endif
