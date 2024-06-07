#include <deal.II/base/function.h>
#include "initial_condition_function.h"
// For initial conditions which need to refer to physics
#include "physics/physics_factory.h"

namespace PHiLiP {

// =========================================================
// Initial Condition Base Class
// =========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction<dim,nstate,real>
::InitialConditionFunction ()
    : dealii::Function<dim,real>(nstate)//,0.0) // 0.0 denotes initial time (t=0)
{
    // Nothing to do here yet
}

// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::InitialConditionFunction_TaylorGreenVortex (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0], y = point[1], z = point[2];

        if(istate==0) {
            // density
            value = this->density(point);
        }
        if(istate==1) {
            // x-velocity
            value = sin(x)*cos(y)*cos(z);
        }
        if(istate==2) {
            // y-velocity
            value = -cos(x)*sin(y)*cos(z);
        }
        if(istate==3) {
            // z-velocity
            value = 0.0;
        }
        if(istate==4) {
            // pressure
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(cos(2.0*z)+2.0);
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 3) {
        std::array<real,nstate> soln_primitive;

        soln_primitive[0] = primitive_value(point,0);
        soln_primitive[1] = primitive_value(point,1);
        soln_primitive[2] = primitive_value(point,2);
        soln_primitive[3] = primitive_value(point,3);
        soln_primitive[4] = primitive_value(point,4);

        const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }

    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TaylorGreenVortex<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::density(const dealii::Point<dim,real> &/*point*/) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    // density
    value = 1.0;
    return value;
}

// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition (Isothermal density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real>
::InitialConditionFunction_TaylorGreenVortex_Isothermal (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>(param)
{}

template <int dim, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real>
::density(const dealii::Point<dim,real> &point) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    // density
    value = this->primitive_value(point, 4); // get pressure
    value *= this->gamma_gas*this->mach_inf_sqr;
    return value;
}

// ========================================================
// 1D BURGERS REWIENSKI -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersRewienski<dim, nstate, real>
::InitialConditionFunction_BurgersRewienski ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersRewienski<dim,nstate,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 1.0;
    return value;
}

// ========================================================
// 1D BURGERS VISCOUS -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersViscous<dim,nstate,real>
::InitialConditionFunction_BurgersViscous ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersViscous<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 0;
    if(point[0] >= 0 && point[0] <= 0.25){
        value = sin(4*dealii::numbers::PI*point[0]);
    }
    return value;

}

// ========================================================
// 1D BURGERS Inviscid -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersInviscid<dim,nstate,real>
::InitialConditionFunction_BurgersInviscid ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersInviscid<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= cos(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= cos(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= cos(dealii::numbers::PI*point[2]);

    return value;
}

// ========================================================
// 1D BURGERS Inviscid Energy-- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_BurgersInviscidEnergy<dim,nstate,real>
::InitialConditionFunction_BurgersInviscidEnergy ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_BurgersInviscidEnergy<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(dealii::numbers::PI*point[2]);

    value += 0.01;
    return value;
}

// ========================================================
// Advection -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_AdvectionEnergy<dim,nstate,real>
::InitialConditionFunction_AdvectionEnergy ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_AdvectionEnergy<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= exp(-20.0*point[0]*point[0]);
    if constexpr(dim >= 2)
        value *= exp(-20.0*point[1]*point[1]);
    if constexpr(dim == 3)
        value *= exp(-20.0*point[2]*point[2]);

    return value;
}

// ========================================================
// Advection OOA -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Advection<dim,nstate,real>
::InitialConditionFunction_Advection()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_Advection<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(2.0*dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(2.0*dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(2.0*dealii::numbers::PI*point[2]);

    return value;
}

// ========================================================
// Convection_diffusion -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ConvDiff<dim,nstate,real>
::InitialConditionFunction_ConvDiff ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_ConvDiff<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(dealii::numbers::PI*point[2]);

    return value;
}

// ========================================================
// Convection_diffusion Energy -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_ConvDiffEnergy<dim,nstate,real>
::InitialConditionFunction_ConvDiffEnergy ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_ConvDiffEnergy<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 1.0;
    if constexpr(dim >= 1)
        value *= sin(dealii::numbers::PI*point[0]);
    if constexpr(dim >= 2)
        value *= sin(dealii::numbers::PI*point[1]);
    if constexpr(dim == 3)
        value *= sin(dealii::numbers::PI*point[2]);

    value += 0.1;

    return value;
}

// ========================================================
// 1D SINE -- Initial Condition for advection_explicit_time_study
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_1DSine<dim,nstate,real>
::InitialConditionFunction_1DSine ()
        : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_1DSine<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 0;
    real pi = dealii::numbers::PI;
    if(point[0] >= 0.0 && point[0] <= 2.0){
        value = sin(2*pi*point[0]/2.0);
    }
    return value;
}

// ========================================================
// Inviscid Isentropic Vortex
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_IsentropicVortex<dim,nstate,real>
::InitialConditionFunction_IsentropicVortex(
        Parameters::AllParameters const *const param)
        : InitialConditionFunction<dim,nstate,real>()
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // This test should only be used for Euler
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(param));
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_IsentropicVortex<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Setting constants
    const double pi = dealii::numbers::PI;
    const double gam = 1.4;
    const double M_infty = sqrt(2/gam);
    const double R = 1;
    const double sigma = 1;
    const double beta = M_infty * 5 * sqrt(2.0)/4.0/pi * exp(1.0/2.0);
    const double alpha = pi/4; //rad

    // Centre of the vortex  at t=0
    const double x0 = 0.0;
    const double y0 = 0.0;
    const double x = point[0] - x0;
    const double y = point[1] - y0;

    const double Omega = beta * exp(-0.5/sigma/sigma* (x/R * x/R + y/R * y/R));
    const double delta_Ux = -y/R * Omega;
    const double delta_Uy =  x/R * Omega;
    const double delta_T  = -(gam-1.0)/2.0 * Omega * Omega;

    // Primitive
    std::array<real,nstate> soln_primitive;
    soln_primitive[0] = pow((1 + delta_T), 1.0/(gam-1.0));
    soln_primitive[1] = M_infty * cos(alpha) + delta_Ux;
    soln_primitive[2] = M_infty * sin(alpha) + delta_Uy;
    #if PHILIP_DIM==3
    soln_primitive[3] = 0;
    #endif
    soln_primitive[nstate-1] = 1.0/gam*pow(1+delta_T, gam/(gam-1.0));

    const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
    return soln_conservative[istate];
}

// ========================================================
// KELVIN-HELMHOLTZ INSTABILITY
// See Chan et al., On the entropy projection..., 2022, Pg. 15
//     Note that some equations are not typed correctly
//     See github.com/trixi-framework/paper-2022-robustness-entropy-projection
//     for initial condition which is implemented herein
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_KHI<dim,nstate,real>
::InitialConditionFunction_KHI (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , atwood_number(param->flow_solver_param.atwood_number)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // This test should only be used for Euler
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(param));
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_KHI<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    const double pi = dealii::numbers::PI;
    
    const double B = 0.5 * (tanh(15*point[1] + 7.5) - tanh(15*point[1] - 7.5));

    const double rho1 = 0.5;
    const double rho2 = rho1 * (1 + atwood_number) / (1 - atwood_number);

    std::array<real,nstate> soln_primitive;
    soln_primitive[0] = rho1 + B * (rho2-rho1);
    soln_primitive[nstate-1] = 1;
    soln_primitive[1] = B - 0.5;
    soln_primitive[2] = 0.1 * sin(2 * pi * point[0]);

    const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
    return soln_conservative[istate];
}

// ========================================================
// ZERO INITIAL CONDITION
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Zero<dim,nstate,real>
::InitialConditionFunction_Zero()
    : InitialConditionFunction<dim,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_Zero<dim, nstate, real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    return 0.0;
}

// ========================================================
// Acoustic Wave (Air) -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_AcousticWave_Air<dim,nstate,real>
::InitialConditionFunction_AcousticWave_Air (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_AcousticWave_Air<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0], y = point[1];

        if(istate==0) {
            // density
            value = 1.0;
        }
        if(istate==1) {
            // x-velocity
            // value = sin(x)*cos(y);
            value = 0.0;
        }
        if(istate==2) {
            // y-velocity
            // value = -cos(x)*sin(y);
            value = 0.0;
        }
        if(istate==3) {
            // pressure
            // value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(2.0);
            // value = 1.0 + (x-x+y-y);
            double const sigma = 0.5;
            double const pi = 6.28318530717958623200 / 2; // pi
            double const mu = 6.28318530717958623200 / 2 ; //max of x
            double fx = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((x-mu)*(x-mu))/(2.0*(sigma*sigma)));
            double fy = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((y-mu)*(y-mu))/(2.0*(sigma*sigma)));
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + fx*fy;
            value = value*3.0/3.0; // chnege this if you want to vary initial temperature 
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_AcousticWave_Air<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 2) {
        std::array<real,nstate> soln_primitive;

        soln_primitive[0] = primitive_value(point,0);
        soln_primitive[1] = primitive_value(point,1);
        soln_primitive[2] = primitive_value(point,2);
        soln_primitive[3] = primitive_value(point,3);

        const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_AcousticWave_Air<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// Acoustic Wave (Species) -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_AcousticWave_Species<dim,nstate,real>
::InitialConditionFunction_AcousticWave_Species (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::inviscid_real_gas;
    this->inviscid_real_gas_physics = std::dynamic_pointer_cast<Physics::InviscidRealGas<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_AcousticWave_Species<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0], y = point[1];

        if(istate==0) {
            // density
            value = 1.0;
        }
        if(istate==1) {
            // x-velocity
            // value = sin(x)*cos(y);
            value = 0.0;
        }
        if(istate==2) {
            // y-velocity
            // value = -cos(x)*sin(y);
            value = 0.0;
        }
        if(istate==3) {
            // pressure
            // value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(2.0);
            // value = 1.0 + (x-x+y-y);
            double const sigma = 0.5;
            double const pi = 6.28318530717958623200 / 2; // pi
            double const mu = 6.28318530717958623200 / 2 ; //max of x
            double fx = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((x-mu)*(x-mu))/(2.0*(sigma*sigma)));
            double fy = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((y-mu)*(y-mu))/(2.0*(sigma*sigma)));
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + fx*fy;
            value = value*3.0/3.0; // chnege this if you want to vary initial temperature 
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_AcousticWave_Species<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 2) {
        std::array<real,nstate> soln_primitive;

        soln_primitive[0] = primitive_value(point,0);
        soln_primitive[1] = primitive_value(point,1);
        soln_primitive[2] = primitive_value(point,2);
        soln_primitive[3] = primitive_value(point,3);

        const std::array<real,nstate> soln_conservative = this->inviscid_real_gas_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_AcousticWave_Species<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// Acoustic Wave (Multi Species) -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_AcousticWave_MultiSpecies<dim,nstate,real>
::InitialConditionFunction_AcousticWave_MultiSpecies (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,dim+2+2-1,double>>( // TO DO: N_SPECIES
                Physics::PhysicsFactory<dim,dim+2+2-1,double>::create_Physics(&parameters_euler)); // TO DO: N_SPECIES
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_AcousticWave_MultiSpecies<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0], y = point[1];

        if(istate==0) {
            // mixture density
            value = 1.00;
        }
        if(istate==1) {
            // x-velocity
            // value = sin(x)*cos(y);
            value = 0.0;
        }
        if(istate==2) {
            // y-velocity
            // value = -cos(x)*sin(y);
            value = 0.0;
        }
        if(istate==3) {
            // pressure
            // value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(2.0);
            // value = 1.0 + (x-x+y-y);
            double const sigma = 0.5;
            double const pi = 6.28318530717958623200 / 2; // pi
            double const mu = 6.28318530717958623200 / 2 ; //max of x
            double fx = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((x-mu)*(x-mu))/(2.0*(sigma*sigma)));
            double fy = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((y-mu)*(y-mu))/(2.0*(sigma*sigma)));
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + fx*fy;
            value = value*3.0/3.0; // chnege this if you want to vary initial temperature 
        }
        if(istate==4){
            // other species density (N2)
            value = 0.79;
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_AcousticWave_MultiSpecies<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 2) {
        std::array<real,nstate> soln_primitive;

        for (int i=0; i<nstate; i++)
        {
            soln_primitive[i] = primitive_value(point,i);
        }

        const std::array<real,nstate> soln_conservative = this->real_gas_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_AcousticWave_MultiSpecies<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D Vortex advection  (Multi Species) -- Initial Condition 
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nstate,real>
::InitialConditionFunction_MultiSpecies_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,dim+2+2-1,double>>( // TO DO: N_SPECIES, dim+2+3-1
                Physics::PhysicsFactory<dim,dim+2+2-1,double>::create_Physics(&parameters_euler)); // TO DO: N_SPECIES, dim+2+3-1
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 1) {
        const real x = point[0];
        const real x_0 = 5.0;
        const real r = sqrt((x-x_0)*(x-x_0));
        const real T_0 = 300.0; // [K]
        const real big_gamma = 50.0;
        const real gamma_0 = 1.4;
        const real y_H2_0 = 0.01277;
        // const real y_O2_0 = 0.101;
        const real a_1 = 0.005;
        // const real a_2 = 0.03;
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);
        const real y_O2 = 1.0 - y_H2;
        // const real y_O2 = (y_O2_0 - a_2*coeff*exp);
        // const real y_N2 = 1.0 - y_H2 - y_O2;

        const std::array Rs = this->real_gas_physics->compute_Rs(this->real_gas_physics->Ru);
        const real R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*this->real_gas_physics->R_ref;
        const real density = pressure/(R_mixture*temperature);

        // dimnsionalized above, non-dimensionalized below
        if(istate==0) {
            // mixture density
            value = density / this->real_gas_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==2) {
            // pressure
            value = pressure / (this->real_gas_physics->density_ref*this->real_gas_physics->u_ref_sqr);
        }
        if(istate==3){
            // other species density (N2)
            value = y_H2;
        }
        // if(istate==4){
        //     // other species density (O2)
        //     value = y_O2;
        // }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 1) {
        std::array<real,nstate> soln_primitive;

        for (int i=0; i<nstate; i++)
        {
            soln_primitive[i] = primitive_value(point,i);
        }

        const std::array<real,nstate> soln_conservative = this->real_gas_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_VortexAdvection<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D Vortex advection  (Euler) -- Initial Condition 
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Euler_VortexAdvection<dim,nstate,real>
::InitialConditionFunction_Euler_VortexAdvection (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_Euler_VortexAdvection<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 1) {
        const real x = point[0];
        const real x_0 = 5.0;
        const real r = sqrt((x-x_0)*(x-x_0));
        const real T_0 = 300.0; // [K]
        const real big_gamma = 50.0;
        const real gamma_0 = 1.4;
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real density = pressure/(this->euler_physics->R_Air_Dim*temperature);

        // dimnsionalized above, non-dimensionalized below
        if(istate==0) {
            // mixture density
            value = density / this->euler_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->euler_physics->u_ref;
        }
        if(istate==2) {
            // pressure
            value = pressure / (this->euler_physics->density_ref*this->euler_physics->u_ref_sqr);
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_Euler_VortexAdvection<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 1) {
        std::array<real,nstate> soln_primitive;

        soln_primitive[0] = primitive_value(point,0);
        soln_primitive[1] = primitive_value(point,1);
        soln_primitive[2] = primitive_value(point,2);

        const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_Euler_VortexAdvection<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D Vortex advection (MS-CP Euler) -- Initial Condition 
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nstate,real>
::InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::multi_species_calorically_perfect_euler;
    this->multi_species_calorically_perfect_euler_physics = std::dynamic_pointer_cast<Physics::MultiSpeciesCaloricallyPerfect<dim,dim+2+3-1,double>>( // TO DO: N_SPECIES, dim+2+3-1
                Physics::PhysicsFactory<dim,dim+2+3-1,double>::create_Physics(&parameters_euler)); // TO DO: N_SPECIES, dim+2+3-1
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 1) {
        const real x = point[0];
        const real x_0 = 5.0;
        const real r = sqrt((x-x_0)*(x-x_0));
        const real T_0 = 300.0; // [K]
        const real big_gamma = 50.0;
        const real gamma_0 = 1.4;
        const real y_H2_0 = 0.01277;
        const real y_O2_0 = 0.101;
        const real a_1 = 0.005;
        const real a_2 = 0.03;
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);
        const real y_O2 = (y_O2_0 - a_2*coeff*exp);
        const real y_N2 = 1.0 - y_H2 - y_O2;

        const std::array Rs = this->multi_species_calorically_perfect_euler_physics->compute_Rs(this->multi_species_calorically_perfect_euler_physics->Ru);
        const real R_mixture = (y_H2*Rs[0] + y_O2*Rs[1] + y_N2*Rs[2])*this->multi_species_calorically_perfect_euler_physics->R_ref;
        const real density = pressure/(R_mixture*temperature);

        // dimnsionalized above, non-dimensionalized below
        if(istate==0) {
            // mixture density
            value = density / this->multi_species_calorically_perfect_euler_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->multi_species_calorically_perfect_euler_physics->u_ref;
        }
        if(istate==2) {
            // pressure
            value = pressure / (this->multi_species_calorically_perfect_euler_physics->density_ref*this->multi_species_calorically_perfect_euler_physics->u_ref_sqr);
        }
        if(istate==3){
            // other species density (N2)
            value = y_H2;
        }
        if(istate==4){
            // other species density (O2)
            value = y_O2;
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 1) {
        std::array<real,nstate> soln_primitive;

        for (int i=0; i<nstate; i++)
        {
            soln_primitive[i] = primitive_value(point,i);
        }

        const std::array<real,nstate> soln_conservative = this->multi_species_calorically_perfect_euler_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D Bubble advection  (Euler) -- Initial Condition 
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_Euler_BubbleAdvection<dim,nstate,real>
::InitialConditionFunction_Euler_BubbleAdvection (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}
template <int dim, int nstate, typename real>
real InitialConditionFunction_Euler_BubbleAdvection<dim,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 1) {
        const real relax = 0.75;

        const real x = point[0];
        const real xi = 10.0/2.0;
        const real T_0 = 300.0; // [K]
        const real theta = 2.0;

        const real pressure = 100000; // [N/m^2] -> [bar]
        const real velocity = 50.0/2.0; // [m/s]
        const real temperature = 0.5*((1.0+theta) +(1.0-theta)*tanh(relax*(abs(x)-xi)))*T_0; // [K]
        const real density = pressure/(this->euler_physics->R_Air_Dim*temperature);

        // dimnsionalized above, non-dimensionalized below
        if(istate==0) {
            // mixture density
            value = density / this->euler_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->euler_physics->u_ref;
        }
        if(istate==2) {
            // pressure
            value = pressure / (this->euler_physics->density_ref*this->euler_physics->u_ref_sqr);
        }
    }
    return value;
}

template <int dim, int nstate, typename real>
real InitialConditionFunction_Euler_BubbleAdvection<dim,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 1) {
        std::array<real,nstate> soln_primitive;

        soln_primitive[0] = primitive_value(point,0);
        soln_primitive[1] = primitive_value(point,1);
        soln_primitive[2] = primitive_value(point,2);

        const std::array<real,nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
        value = soln_conservative[istate];
    }
    return value;
}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_Euler_BubbleAdvection<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// =========================================================
// Initial Condition Factory
// =========================================================
template <int dim, int nstate, typename real>
std::shared_ptr<InitialConditionFunction<dim, nstate, real>>
InitialConditionFactory<dim,nstate, real>::create_InitialConditionFunction(
    Parameters::AllParameters const *const param)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = param->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex) {
        if constexpr (dim==3 && nstate==dim+2){ 
            // Get the density initial condition type
            const DensityInitialConditionEnum density_initial_condition_type = param->flow_solver_param.density_initial_condition_type;
            if(density_initial_condition_type == DensityInitialConditionEnum::uniform) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,nstate,real> >(
                        param);
            } else if (density_initial_condition_type == DensityInitialConditionEnum::isothermal) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real> >(
                        param);
            }
        }
    } else if (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence) {
        if constexpr (dim==3 && nstate==dim+2) return nullptr; // nullptr since DHIT case initializes values from file
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersRewienski<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersViscous<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::naca0012 || flow_type == FlowCaseEnum::gaussian_bump) {
        if constexpr ((dim==2 || dim==3) && nstate==dim+2) {
            Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
                    param,
                    param->euler_param.ref_length,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf,
                    param->euler_param.angle_of_attack,
                    param->euler_param.side_slip_angle);
            return std::make_shared<FreeStreamInitialConditions<dim,nstate,real>>(euler_physics_double);
        }
    } else if (flow_type == FlowCaseEnum::burgers_inviscid && param->use_energy==false) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersInviscid<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::burgers_inviscid && param->use_energy==true) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersInviscidEnergy<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::advection && param->use_energy==true) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_AdvectionEnergy<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::advection && param->use_energy==false) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_Advection<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::convection_diffusion && !param->use_energy) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_ConvDiff<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::convection_diffusion && param->use_energy) {
        return std::make_shared<InitialConditionFunction_ConvDiffEnergy<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::periodic_1D_unsteady) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_1DSine<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::isentropic_vortex) {
        if constexpr (dim>1 && nstate==dim+2) return std::make_shared<InitialConditionFunction_IsentropicVortex<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::kelvin_helmholtz_instability) {
        if constexpr (dim>1 && nstate==dim+2) return std::make_shared<InitialConditionFunction_KHI<dim,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::non_periodic_cube_flow) {
        if constexpr (dim==2 && nstate==1)  return std::make_shared<InitialConditionFunction_Zero<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::acoustic_wave_air) {
        if constexpr (dim==2 && nstate==dim+2){ 
            return std::make_shared<InitialConditionFunction_AcousticWave_Air<dim,nstate,real> >(param);
        }
    } else if (flow_type == FlowCaseEnum::acoustic_wave_species) {
        if constexpr (dim==2 && nstate==dim+2){ 
            return std::make_shared<InitialConditionFunction_AcousticWave_Species<dim,nstate,real> >(param);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_acoustic_wave) {
        if constexpr (dim==2 && nstate==dim+2+2-1){ // TO DO: N_SPECIES
            return std::make_shared<InitialConditionFunction_AcousticWave_MultiSpecies<dim,nstate,real> >(param);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_vortex_advection) {
        if constexpr (dim==1 && nstate==dim+2+2-1){ // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
            return std::make_shared<InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nstate,real> >(param);
        }
    } else if (flow_type == FlowCaseEnum::euler_vortex_advection) {
        if constexpr (dim==1 && nstate==dim+2){ 
            return std::make_shared<InitialConditionFunction_Euler_VortexAdvection<dim,nstate,real> >(param);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection) {
        if constexpr (dim==1 && nstate==dim+2+3-1){ 
            return std::make_shared<InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nstate,real> >(param);
        }
    } else if (flow_type == FlowCaseEnum::euler_bubble_advection) {
        if constexpr (dim==1 && nstate==dim+2){ 
            return std::make_shared<InitialConditionFunction_Euler_BubbleAdvection<dim,nstate,real> >(param);
        }        
    } else {
        std::cout << "Invalid Flow Case Type. You probably forgot to add it to the list of flow cases in initial_condition_function.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

template class InitialConditionFunction <PHILIP_DIM, 1, double>;
template class InitialConditionFunction <PHILIP_DIM, 2, double>;
template class InitialConditionFunction <PHILIP_DIM, 3, double>;
template class InitialConditionFunction <PHILIP_DIM, 4, double>;
template class InitialConditionFunction <PHILIP_DIM, 5, double>;
template class InitialConditionFunction <PHILIP_DIM, 6, double>;

template class InitialConditionFactory <PHILIP_DIM, 1, double>;
template class InitialConditionFactory <PHILIP_DIM, 2, double>;
template class InitialConditionFactory <PHILIP_DIM, 3, double>;
template class InitialConditionFactory <PHILIP_DIM, 4, double>;
template class InitialConditionFactory <PHILIP_DIM, 5, double>;
template class InitialConditionFactory <PHILIP_DIM, 6, double>;
 
#if PHILIP_DIM==1
template class InitialConditionFunction_BurgersViscous <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_BurgersRewienski <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_BurgersInviscid <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_BurgersInviscidEnergy <PHILIP_DIM, 1, double>;
#endif

#if PHILIP_DIM==3
template class InitialConditionFunction_TaylorGreenVortex <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_TaylorGreenVortex_Isothermal <PHILIP_DIM, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM>1
template class InitialConditionFunction_IsentropicVortex <PHILIP_DIM, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM==2
template class InitialConditionFunction_KHI <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_AcousticWave_Air <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFunction_AcousticWave_Species <PHILIP_DIM, PHILIP_DIM+2, double>;
#endif

// functions instantiated for all dim
template class InitialConditionFunction_Zero <PHILIP_DIM,1, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,2, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,3, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,4, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,5, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,6, double>;
template class InitialConditionFunction_Advection <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_AdvectionEnergy <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_ConvDiff <PHILIP_DIM, 1, double>;
template class InitialConditionFunction_ConvDiffEnergy <PHILIP_DIM,1,double>;

} // PHiLiP namespace
