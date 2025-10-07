#include <deal.II/base/function.h>
#include "initial_condition_function.h"
// For initial conditions which need to refer to physics
#include "physics/physics_factory.h"

namespace PHiLiP {

// =========================================================
// Initial Condition Base Class
// =========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction<dim,nspecies,nstate,real>
::InitialConditionFunction ()
    : dealii::Function<dim,real>(nstate)//,0.0) // 0.0 denotes initial time (t=0)
{
    // Nothing to do here yet
}

// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,nspecies,nstate,real>
::InitialConditionFunction_TaylorGreenVortex (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{}
template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nspecies,nstate,real>
::InitialConditionFunction_TaylorGreenVortex_Isothermal (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction_TaylorGreenVortex<dim,nspecies,nstate,real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_BurgersRewienski<dim, nspecies, nstate, real>
::InitialConditionFunction_BurgersRewienski ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_BurgersRewienski<dim,nspecies,nstate,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 1.0;
    return value;
}

// ========================================================
// 1D BURGERS VISCOUS -- Initial Condition
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_BurgersViscous<dim,nspecies,nstate,real>
::InitialConditionFunction_BurgersViscous ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_BurgersViscous<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_BurgersInviscid<dim,nspecies,nstate,real>
::InitialConditionFunction_BurgersInviscid ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_BurgersInviscid<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_BurgersInviscidEnergy<dim,nspecies,nstate,real>
::InitialConditionFunction_BurgersInviscidEnergy ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_BurgersInviscidEnergy<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_AdvectionEnergy<dim,nspecies,nstate,real>
::InitialConditionFunction_AdvectionEnergy ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_AdvectionEnergy<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_Advection<dim,nspecies,nstate,real>
::InitialConditionFunction_Advection()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_Advection<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_ConvDiff<dim,nspecies,nstate,real>
::InitialConditionFunction_ConvDiff ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_ConvDiff<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_ConvDiffEnergy<dim,nspecies,nstate,real>
::InitialConditionFunction_ConvDiffEnergy ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_ConvDiffEnergy<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_1DSine<dim,nspecies,nstate,real>
::InitialConditionFunction_1DSine ()
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_1DSine<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_IsentropicVortex<dim,nspecies,nstate,real>
::InitialConditionFunction_IsentropicVortex(
        Parameters::AllParameters const *const param)
        : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // This test should only be used for Euler
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2,double>::create_Physics(param));
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_IsentropicVortex<dim,nspecies,nstate,real>
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
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_KHI<dim,nspecies,nstate,real>
::InitialConditionFunction_KHI (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , atwood_number(param->flow_solver_param.atwood_number)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // This test should only be used for Euler
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2,double>::create_Physics(param));
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_KHI<dim,nspecies,nstate,real>
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
// Initial Condition - Euler Base
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>
::InitialConditionFunction_EulerBase(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction<dim, nspecies, nstate, real>()
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2,double>::create_Physics(&parameters_euler));
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    std::array<real, nstate> soln_primitive;

    soln_primitive[0] = primitive_value(point, 0);
    soln_primitive[1] = primitive_value(point, 1);
    soln_primitive[2] = primitive_value(point, 2);
    
    if constexpr (dim > 1)
        soln_primitive[3] = primitive_value(point, 3);
    if constexpr (dim > 2)
        soln_primitive[4] = primitive_value(point, 4);

    const std::array<real, nstate> soln_conservative = this->euler_physics->convert_primitive_to_conservative(soln_primitive);
    value = soln_conservative[istate];

    return value;
}

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>
::value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point, istate);
    return value;
}

// ========================================================
// 1D Sod Shock tube -- Initial Condition
// See Chen & Shu, Entropy stable high order..., 2017, Pg. 25
// 2D and 3D can be run by extruding grid in those directions
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_SodShockTube<dim,nspecies,nstate,real>
::InitialConditionFunction_SodShockTube (
        Parameters::AllParameters const* const param)
        : InitialConditionFunction_EulerBase<dim,nspecies,nstate,real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_SodShockTube<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 1 && nstate == (dim+2)) {
        const real x = point[0];
        if (x < 0) {
            if (istate == 0) {
                // density
                value = 1.0;
            }
            if (istate == nstate - 1) {
                // pressure
                value = 1.0;
            }
        } else {
            if (istate == 0) {
                // density
                value = 0.125;
            }
            if (istate == nstate - 1) {
                // pressure
                value = 0.1;
            }
        }
    }

    return value;
}

// ========================================================
// 1D Leblanc Shock tube -- Initial Condition
// See Zhang & Shu, On positivity-preserving..., 2010 Pg. 14
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_LeblancShockTube<dim,nspecies,nstate,real>
::InitialConditionFunction_LeblancShockTube(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_LeblancShockTube<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 1 && nstate == (dim + 2)) {
        const real x = point[0];
        if (x < 0) {
            if (istate == 0) {
                // density
                value = 2.0;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // pressure
                value = pow(10.0, 9.0);
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 0.001;
            }
            if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            if (istate == 2) {
                // pressure
                value = 1.0;
            }
        }
    }
    return value;
}

// ========================================================
// 1D Shu-Osher Problem -- Initial Condition
// See Johnsen et al., Assessment of high-resolution..., 2010 Pg. 7
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_ShuOsherProblem<dim, nspecies, nstate, real>
::InitialConditionFunction_ShuOsherProblem(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_ShuOsherProblem<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 1 && nstate == (dim + 2)) {
        const real x = point[0];
        if (x < -4) {
            if (istate == 0) {
                // density
                value = 3.857143;
            }
            else if (istate == 1) {
                // x-velocity
                value = 2.629369;
            }
            else if (istate == 2) {
                // pressure
                value = 10.33333;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 1 + 0.2 * sin(5 * x);
            }
            else if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            else if (istate == 2) {
                // pressure
                value = 1.0;
            }
        }
    }
    return value;
}

// =====================================================================
// Low Density Euler -- Initial Condition
// See Dzanic & Martinelli, High-order limiting..., 2025, Pg. 15
// =====================================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_LowDensity<dim,nspecies,nstate,real>
::InitialConditionFunction_LowDensity(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_LowDensity<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 1 && nstate == (dim + 2)) {
        const real x = point[0];
        if (istate == 0) {
            // density
            value = 0.01 + exp(-500.0*pow(x,2.0));
        }
        else {
            value = 1.0;
        }
    }

    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];

        if (istate == 0) {
            // density
            value = 0.01 + exp(-500.0*(pow(x, 2.0)+pow(y, 2.0)));
        }
        else {
            // x-velocity
            value = 1.0;
        }
    }
    return value;
}

// ==================================================================
// Double Mach Reflection Problem (2D) -- Initial Condition
// See Lin, Chan, and Tomas. "A positivity preserving ...", 2023, p20
// ==================================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_DoubleMachReflection<dim, nspecies, nstate, real>
::InitialConditionFunction_DoubleMachReflection(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_DoubleMachReflection<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        const real x = point[0];
        const real y = point[1];
        if (y > sqrt(3)*(x - (1.0/6.0))) {
            if (istate == 0) {
                // density
                value = 8.0;
            }
            else if (istate == 1) {
                // x-velocity
                value = 33.0*sqrt(3.0)/8.0;
            }
            else if (istate == 2) {
                // y-velocity
                value = -33.0/8.0;
            }
            else if (istate == 3) {
                // pressure
                value = 116.5;
            }
        }
        else {
            if (istate == 0) {
                // density
                value = 1.4;
            }
            else if (istate == 1) {
                // x-velocity
                value = 0.0;
            }
            else if (istate == 2) {
                // y-velocity
                value = 0.0;
            }
            else if (istate == 3) {
                // pressure
                value = 1.0;
            }
        }
    }
    return value;
}

// ========================================================
// Shock Diffraction (backwards facing step) (2D) -- Initial Condition
// See Zhang & Shu, On positivity-preserving..., 2010 Pg. 15
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_ShockDiffraction<dim, nspecies, nstate, real>
::InitialConditionFunction_ShockDiffraction(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_ShockDiffraction<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    const real x = point[0];
    //real y = point[1];
    if constexpr (dim == 2 && nstate == (dim + 2)) {
        if (x <= 0.5) {
            if (istate == 0) {
                // density
                value = 7.041132906907898;
            }
            else if (istate == 1) {
                // x-velocity
                value = 4.07794695481336;
            }
            else if (istate == 2) {
                // y-velocity
                value = 0.0;
            }
            else if (istate == 3) {
                // pressure
                value = 30.05945;
            }
        }
        else {
           if (istate == 0) {
               // density
               value = 1.4;
           }
           else if (istate == 1) {
               // x-velocity
               value = 0.0;
           }
           else if (istate == 2) {
               // y-velocity
               value = 0.0;
           }
           else if (istate == 3) {
               // pressure
               value = 1.0;
           }
        }
    }
    return value;
}


// ========================================================
// Astrophysical Mach Jet (2D) -- Initial Condition
// See Zhang & Shu, On positivity-preserving..., 2010 Pg. 14
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_AstrophysicalJet<dim, nspecies, nstate, real>
::InitialConditionFunction_AstrophysicalJet(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_AstrophysicalJet<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& /*point*/, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr (dim == 2 && nstate == (dim + 2)) {

        if (istate == 0) {
            // density
            value = 0.5;
        }
        else if (istate == 1) {
            // x-velocity
            value = 0.0;
        }
        else if (istate == 2) {
            // y-velocity
            value = 0.0;
        }
        else if (istate == 3) {
            // pressure
            value = 0.4127;
        }
    }
    return value;
}


// ========================================================
// Strong Vortex Shock Wave Interaction (2D) -- Initial Condition
// See High Fidelity CFD Workshop 2022
// Unsteady Supersonic/Hypersonic Test Suite
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_SVSW<dim, nspecies, nstate, real>
::InitialConditionFunction_SVSW(
    Parameters::AllParameters const* const param)
    : InitialConditionFunction_EulerBase<dim, nspecies, nstate, real>(param)
{}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_SVSW<dim, nspecies, nstate, real>
::primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate) const
{
    real value = 0.0;
    const real x = point[0];
    const real y = point[1];

    if constexpr (dim == 2 && nstate == (dim + 2)) {
        // Ideal gas
        const real gamma = 1.4;
        const real R = 1.0;

        // Upstream conditions
        const real rho_u = 1.0;
        const real u_u = 1.5*sqrt(1.4);
        const real v_u = 0.0;
        const real p_u = 1.0;
        const real t_u = p_u/(rho_u*R);



        // Shock condition
        const real M_s = 1.5;

        // Downstream conditions
        const real rho_d = (rho_u * (gamma + 1.0) * M_s * M_s) / (2.0 + (gamma - 1.0) * M_s * M_s);
        const real u_d = (u_u * (2.0 + ((gamma - 1.0) * M_s * M_s)))/((gamma + 1.0) * M_s * M_s);
        const real v_d = 0.0;
        const real p_d = p_u * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (M_s * M_s - 1.0));

        if (x <= 0.5){
            if (istate == 0) {
                // density
                value = rho_u;
            }
            else if (istate == 1) {
                // x-velocity
                value = u_u;
            }
            else if (istate == 2) {
                // y-velocity
                value = v_u;
            }
            else if (istate == 3) {
                // pressure
                value = p_u;
            }
        } else {
            if (istate == 0) {
                // density
                value = rho_d;
            }
            else if (istate == 1) {
                // x-velocity
                value = u_d;
            }
            else if (istate == 2) {
                // y-velocity
                value = v_d;
            }
            else if (istate == 3) {
                // pressure
                value = p_d;
            }            
        }

        if(x <= 0.5) {
            // Vortex location
            const real x_c = 0.25; const real y_c = 0.5;

            // Vortex sizes
            const real a = 0.075; const real b = 0.175;

            // Vortex strength
            const real M_v = 0.9; const real v_m = M_v * sqrt(gamma);

            // Distance from vortex
            const real dx = x - x_c;
            const real dy = y - y_c;
            const real r = sqrt((dx*dx) + (dy*dy));

            real temperature = 0.0;

            // Superimpose vortex
            if (r<=b) {
                const double sin_theta = dy/r;
                const double cos_theta = dx/r;

                if (r<=a) {
                    const real mag = v_m * r / a;
                    if(istate == 1)
                        value = u_u - mag*sin_theta;
                    else if(istate == 2)
                        value = v_u + mag*cos_theta;
                    else {
                        // Temperature at a, integrated from ODE
                        real radial_term = -2.0 * b * b * log(b) - (0.5 * a * a) + (2.0 * b * b * log(a)) + (0.5 * b * b * b * b / (a * a));
                        const real t_a = t_u - (gamma - 1.0) * pow(v_m * a / (a * a - b * b), 2.0) * radial_term / (R * gamma);
                        radial_term = 0.5 * (1.0 - r * r / (a * a));
                        temperature = t_a - (gamma - 1.0) * v_m * v_m * radial_term / (R * gamma);
                    } 
                } else {
                    const real mag = v_m * a * (r - b * b / r)/(a * a - b * b);
                    if(istate == 1)
                        value = u_u - mag * sin_theta;
                    else if (istate == 2)
                        value = v_u + mag * cos_theta;
                    else {
                        const real radial_term = -2.0 * b * b * log(b) - (0.5 * r * r) + (2.0 * b * b * log(r)) + (0.5 * b * b * b * b / (r * r));
                        temperature = t_u - (gamma - 1.0) * pow(v_m * a/(a * a - b * b), 2.0) * radial_term / (R * gamma);
                    }
                }

                if (istate == 0)
                    value = rho_u * pow(temperature/t_u, 1.0/(gamma - 1.0));
                else if (istate == 3)
                    value = p_u * pow(temperature/t_u, gamma/(gamma - 1.0));
            }
        }
    }
    return value;
}


// ========================================================
// Acoustic Wave (Multi Species) -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_AcousticWave_MultiSpecies<dim,nspecies,nstate,real>
::InitialConditionFunction_AcousticWave_MultiSpecies (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler));
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_AcousticWave_MultiSpecies<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0], y = point[1];

        if(istate==0) {
            value = 1.00;
        }
        if(istate==1) {
            value = 0.0;
        }
        if(istate==2) {
            value = 0.0;
        }
        if(istate==3) {
            double const sigma = 0.5;
            double const pi = 6.28318530717958623200 / 2; // pi
            double const mu = 6.28318530717958623200 / 2 ; //max of x
            double fx = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((x-mu)*(x-mu))/(2.0*(sigma*sigma)));
            double fy = (1.0/sqrt(2.0*pi*sigma*sigma))*exp(-((y-mu)*(y-mu))/(2.0*(sigma*sigma)));
            value = 1.0/(this->gamma_gas*this->mach_inf_sqr) + fx*fy;
            value = value*3.0/3.0; // chaege this if you want to vary initial temperature 
        }
        if(istate==4){
            // species density (N2)
            value = 0.79;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_AcousticWave_MultiSpecies<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_AcousticWave_MultiSpecies<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D Vortex advection  (Multi Species) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>( 
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); 
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nspecies,nstate,real>
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
        const real a_1 = 0.005;
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);

        const std::array<real,nspecies> Rs = this->real_gas_physics->compute_Rs(this->real_gas_physics->Ru);
        real y_O2;
        real R_mixture;
        // For a 2 species test
        if constexpr(nspecies==2 && nstate==dim+2+nspecies-1) {
            y_O2 = 1.0 - y_H2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*this->real_gas_physics->R_ref;
        }
        // For a 3 species test
        if constexpr(nspecies==3 && nstate==dim+2+nspecies-1) {
            const real y_O2_0 = 0.101;
            const real a_2 = 0.03;
            y_O2 = (y_O2_0 - a_2*coeff*exp);
            const real y_N2 = 1.0 - y_H2 - y_O2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1] + y_N2*Rs[2])*this->real_gas_physics->R_ref;
        }
        const real density = pressure/(R_mixture*temperature);

        // dimensionalized above, non-dimensionalized below
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
        if(istate==4){
            // other species density (O2)
            value = y_O2;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_VortexAdvection<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D Vortex advection  (Multi Species, High-Temperature) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); 
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection<dim,nspecies,nstate,real>
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
        const real a_1 = 0.005;
        const real pi = 6.28318530717958623200 / 2; // pi

        real pressure = 101325; // [N/m^2]
        pressure *= 5.0;
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        temperature *= 5.0;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);

        const std::array Rs = this->real_gas_physics->compute_Rs(this->real_gas_physics->Ru);
        real y_O2;
        real R_mixture;
        // For a 2 species test
        if constexpr(nspecies==2 && nstate==dim+2+nspecies-1) {
            y_O2 = 1.0 - y_H2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*this->real_gas_physics->R_ref;
        }
        // For a 3 species test
        if constexpr(nspecies==3 && nstate==dim+2+nspecies-1) {
            const real y_O2_0 = 0.101;
            const real a_2 = 0.03;
            y_O2 = (y_O2_0 - a_2*coeff*exp);
            const real y_N2 = 1.0 - y_H2 - y_O2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1] + y_N2*Rs[2])*this->real_gas_physics->R_ref;
        }
        const real density = pressure/(R_mixture*temperature);

        // dimensionalized above, non-dimensionalized below
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
        if constexpr(nstate==dim+2+3-1) {
            if(istate==4){
            // other species density (O2)
            value = y_O2;
            }
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}


// ========================================================
// 1D Vortex advection (MS-CP Euler) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::multi_species_calorically_perfect_euler;
    this->multi_species_calorically_perfect_euler_physics = std::dynamic_pointer_cast<Physics::MultiSpeciesCaloricallyPerfect<dim,nspecies,dim+2+nspecies-1,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); 
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nspecies,nstate,real>
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
        const real a_1 = 0.005;
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);

        const std::array Rs = this->multi_species_calorically_perfect_euler_physics->compute_Rs(this->multi_species_calorically_perfect_euler_physics->Ru);
        real y_O2;
        real R_mixture;
        // For a 2 species test
        if constexpr(nspecies==2 && nstate==dim+2+nspecies-1) {
            y_O2 = 1.0 - y_H2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*this->multi_species_calorically_perfect_euler_physics->R_ref;
        }
        // For a 3 species test
        if constexpr(nspecies==3 && nstate==dim+2+nspecies-1) {
            const real y_O2_0 = 0.101;
            const real a_2 = 0.03;
            y_O2 = (y_O2_0 - a_2*coeff*exp);
            const real y_N2 = 1.0 - y_H2 - y_O2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1] + y_N2*Rs[2])*this->multi_species_calorically_perfect_euler_physics->R_ref;
        }
        const real density = pressure/(R_mixture*temperature);

        // dimensionalized above, non-dimensionalized below
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
        if constexpr(nspecies==3 && nstate==dim+2+nspecies-1) {
            if(istate==4){
            // other species density (O2)
            value = y_O2;
            }
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// =============================================================
// Isentropic Euler Vortex (Multi Species) -- Initial Condition 
// =============================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_IsentropicEulerVortex<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_IsentropicEulerVortex (
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler));
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_IsentropicEulerVortex<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0];
        const real y = point[1];

        // constant value
        const real x_0 = 0.0;
        const real y_0 = 0.0;
        const real beta = 13.5;
        const real radius = 1.5;
        const real U_0 = 0.0;
        const real V_0 = 1.0;
        const real M = 0.40;
        const real pi = 6.28318530717958623200 / 2; // pi
        const real L = 10.0;
        const real alpha_N2 = 0.50*sin(pi/L*(x-x_0))+0.50;
        const real alpha_O2 = 1.0 - alpha_N2;
        const real mixture_gamma = 1.4;

        const real f = (1.0 - (x-x_0)*(x-x_0) - (y-y_0)*(y-y_0)) / (2.0*radius*radius);
        const real density_N2 = alpha_N2*pow( (1.0 - ((mixture_gamma-1.0)*beta*beta*M*M/(8.0*pi*pi))*exp(2.0*f)), 1.0/(mixture_gamma-1.0) );
        const real density_O2 = alpha_O2*pow( (1.0 - ((mixture_gamma-1.0)*beta*beta*M*M/(8.0*pi*pi))*exp(2.0*f)), 1.0/(mixture_gamma-1.0) );
        const real mixture_density = density_N2 + density_O2;
        const real u = U_0 + beta*y/(2.0*pi*radius)*exp(f);
        const real v = V_0 - beta*x/(2.0*pi*radius)*exp(f);

        const real temperature_modification = 2.0;
        const real mixture_pressure = 1.0/(mixture_gamma*M*M)*pow(mixture_density,mixture_gamma) * temperature_modification;

        // non-dimensionalized values above, non-dimensionalized values below
        if(istate==0) {
            // mixture density
            value = mixture_density;
        }
        if(istate==1) {
            // x-velocity
            value = u;
        }
        if(istate==2) {
            // y-velocity
            value = v;
        }
        if(istate==3) {
            // pressure
            value = mixture_pressure;
        }
        if(istate==4){
            // other species density (N2)
            value = density_N2/mixture_density;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_IsentropicEulerVortex<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_IsentropicEulerVortex<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 2D Vortex advection  (Multi Species) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note: Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(    // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0];
        const real y = point[1];

        const real x_0 = 5.0;
        const real y_0 = 5.0;
        const real r = sqrt((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0));
        const real T_0 = 300.0; // [K]
        const real big_gamma = 50.0;
        const real gamma_0 = 1.4;
        const real y_H2_0 = 0.01277;
        const real a_1 = 0.005; 
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);

        const std::array Rs = this->real_gas_physics->compute_Rs(this->real_gas_physics->Ru);
        real y_O2;
        real R_mixture;
        // For a 2 species test
        if constexpr(nspecies==2 && nstate==dim+2+nspecies-1) {
            y_O2 = 1.0 - y_H2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*this->real_gas_physics->R_ref;
        }
        const real density = pressure/(R_mixture*temperature);

        // dimensionalized values above, non-dimensionalized values below
        if(istate==0) {
            // mixture density
            value = density / this->real_gas_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==2) {
            // y-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==3) {
            // pressure
            value = pressure / (this->real_gas_physics->density_ref*this->real_gas_physics->u_ref_sqr);
        }
        if(istate==4){
            // other species density (N2)
            value = y_H2;
        }
        if(istate==5){
            // other species density (O2)
            value = y_O2;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 2D fuel drop advection  (Multi Species) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_FuelDropAdvection<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_FuelDropAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note that Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>( // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_FuelDropAdvection<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 2) {
        const real x = point[0];
        const real y = point[1];

        const real x_0 = 0.50;
        const real y_0 = 0.50;

        const real r = sqrt((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0));
        real mass_fraction_fuel;
        real mass_fraction_N2;
        const std::array<real,nspecies> Rs = this->real_gas_physics->compute_Rs(this->real_gas_physics->Ru);
        const real velocity = 50.0;

        const real r_0 = 1.0/3.141592653589793;
        real steep;
        if (r<r_0)
        {
            steep = 23.0;
            mass_fraction_fuel = 1.0 - 0.5*(1 + tanh( steep*(r-r_0) ));
        }
        else
        {
            steep = 60.0;
            mass_fraction_fuel = 1.0 - 0.5*(1 + tanh( steep*(r-r_0) ));          
        }
        mass_fraction_N2 = 1.0 - mass_fraction_fuel;
        real R_mixture = 0.0;
        if(nspecies==2)
            R_mixture = (mass_fraction_N2*Rs[0] + mass_fraction_fuel*Rs[1])*this->real_gas_physics->R_ref;
        const real temperature = 573.0;
        const real pressure = 600*100*100;
        const real density = pressure/(R_mixture*temperature);

        // dimensionalized values above, non-dimensionalized values below
        if(istate==0) {
            // mixture density
            value = density / this->real_gas_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==2) {
            // y-velocity
            value = 0.0 / this->real_gas_physics->u_ref;
        }
        if(istate==3) {
            // pressure
            value = pressure / (this->real_gas_physics->density_ref*this->real_gas_physics->u_ref_sqr);
        }
        if(istate==4){
            // species mass fraction density (N2)
            value = mass_fraction_N2;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_FuelDropAdvection<dim,nspecies,nstate,real>
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_FuelDropAdvection<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 3D Vortex advection  (Multi Species) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note: Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(    // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0];
        const real y = point[1];

        const real x_0 = 5.0;
        const real y_0 = 5.0;
        const real r = sqrt((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0));
        const real T_0 = 300.0; // [K]
        const real big_gamma = 50.0;
        const real gamma_0 = 1.4;
        const real y_H2_0 = 0.01277;
        const real a_1 = 0.005; 
        const real pi = 6.28318530717958623200 / 2; // pi

        const real pressure = 101325; // [N/m^2]
        const real velocity = 100.0; // [m/s]
        const real exp = std::exp(0.50*(1-r*r));
        const real coeff = 2*pi/(gamma_0*big_gamma);
        const real temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const real y_H2 = (y_H2_0 - a_1*coeff*exp);

        const std::array Rs = this->real_gas_physics->compute_Rs(this->real_gas_physics->Ru);
        real y_O2;
        real R_mixture;
        // For a 2 species test
        if constexpr(nspecies==2 && nstate==dim+2+nspecies-1) {
            y_O2 = 1.0 - y_H2;
            R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*this->real_gas_physics->R_ref;
        }
        const real density = pressure/(R_mixture*temperature);

        // dimensionalized values above, non-dimensionalized values below
        if(istate==0) {
            // mixture density
            value = density / this->real_gas_physics->density_ref;
        }
        if(istate==1) {
            // x-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==2) {
            // y-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==3) {
            // z-velocity
            value = velocity / this->real_gas_physics->u_ref;
        }
        if(istate==4) {
            // pressure
            value = pressure / (this->real_gas_physics->density_ref*this->real_gas_physics->u_ref_sqr);
        }
        if(istate==5){
            // other species density (N2)
            value = y_H2;
        }
        if(istate==6){
            // other species density (O2)
            value = y_O2;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection<dim,nspecies,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 3) {
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 3D TaylorGreenVortex  (Multi Species, Uniform mass fractions) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_TaylorGreenVortex<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_TaylorGreenVortex(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note: Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(    // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_TaylorGreenVortex<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0];
        const real y = point[1];
        const real z = point[2];

        // dimensionalized values above, non-dimensionalized values below
        if(istate==0) {
            // mixture density
            value = 1.0;
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
        if(istate==5){
            // mass fraction (N2)
            value = 0.79;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_TaylorGreenVortex<dim,nspecies,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 3) {
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_TaylorGreenVortex<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 3D TaylorGreenVortex  (Multi Species, Mixture mass fractions) -- Initial Condition 
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex<dim,nspecies,nstate,real>
::InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex(
        Parameters::AllParameters const *const param)
    : InitialConditionFunction<dim,nspecies,nstate,real>()
    , gamma_gas(param->euler_param.gamma_gas)
    , mach_inf(param->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Euler object; create using dynamic_pointer_cast and the create_Physics factory
    // Note: Euler primitive/conservative vars are the same as NS
    PHiLiP::Parameters::AllParameters parameters_euler = *param;
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::real_gas;
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+2+nspecies-1,double>>(    // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
                Physics::PhysicsFactory<dim,nspecies,dim+2+nspecies-1,double>::create_Physics(&parameters_euler)); // Note: modify this when you change the number of species. nstate == dim+2+(nspecies)-1
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex<dim,nspecies,nstate,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0];
        const real y = point[1];
        const real z = point[2];
        const double pi = 6.28318530717958623200/2.0;
        double mass_fraction_N2;
        if (x > pi && y > pi || x < pi && y < pi) {
            mass_fraction_N2 = 0.90;
        }
        else {
            mass_fraction_N2 = 0.10;
        }

        // dimensionalized values above, non-dimensionalized values below
        if(istate==0) {
            // mixture density
            value = 1.0;
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
        if(istate==5){
            // mass fraction (N2)
            value = mass_fraction_N2;
        }
    }
    return value;
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex<dim,nspecies,nstate,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if constexpr(dim == 3) {
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

template <int dim, int nspecies, int nstate, typename real>
inline real InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// ZERO INITIAL CONDITION
// ========================================================
template <int dim, int nspecies, int nstate, typename real>
InitialConditionFunction_Zero<dim,nspecies,nstate,real>
::InitialConditionFunction_Zero()
    : InitialConditionFunction<dim,nspecies,nstate,real>()
{
    // Nothing to do here yet
}

template <int dim, int nspecies, int nstate, typename real>
real InitialConditionFunction_Zero<dim, nspecies, nstate, real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    return 0.0;
}

// =========================================================
// Initial Condition Factory
// =========================================================
template <int dim, int nspecies, int nstate, typename real>
std::shared_ptr<InitialConditionFunction<dim, nspecies, nstate, real>>
InitialConditionFactory<dim,nspecies,nstate, real>::create_InitialConditionFunction(
    Parameters::AllParameters const *const param)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = param->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex) {
        if constexpr (dim==3 && nstate==dim+2){ 
            // Get the density initial condition type
            const DensityInitialConditionEnum density_initial_condition_type = param->flow_solver_param.density_initial_condition_type;
            if(density_initial_condition_type == DensityInitialConditionEnum::uniform) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,nspecies,nstate,real> >(
                        param);
            } else if (density_initial_condition_type == DensityInitialConditionEnum::isothermal) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nspecies,nstate,real> >(
                        param);
            }
        }
    } else if (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence) {
        if constexpr (dim==3 && nstate==dim+2) return nullptr; // nullptr since DHIT case initializes values from file
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersRewienski<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersViscous<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::naca0012 || flow_type == FlowCaseEnum::gaussian_bump) {
        if constexpr ((dim==2 || dim==3) && nstate==dim+2) {
            Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
                    param,
                    param->euler_param.ref_length,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf,
                    param->euler_param.angle_of_attack,
                    param->euler_param.side_slip_angle);
            return std::make_shared<FreeStreamInitialConditions<dim,nspecies,nstate,real>>(euler_physics_double);
        }
    } else if (flow_type == FlowCaseEnum::burgers_inviscid && param->use_energy==false) {
        if constexpr (nstate==dim && dim<3) return std::make_shared<InitialConditionFunction_BurgersInviscid<dim,nspecies,nstate,real> >();
    } else if (flow_type == FlowCaseEnum::burgers_inviscid && param->use_energy==true) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersInviscidEnergy<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::advection && param->use_energy==true) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_AdvectionEnergy<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::advection && param->use_energy==false) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_Advection<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::convection_diffusion && !param->use_energy) {
        if constexpr (nstate==1) return std::make_shared<InitialConditionFunction_ConvDiff<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::convection_diffusion && param->use_energy) {
        return std::make_shared<InitialConditionFunction_ConvDiffEnergy<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::periodic_1D_unsteady) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_1DSine<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::isentropic_vortex) {
        if constexpr (dim>1 && nstate==dim+2) return std::make_shared<InitialConditionFunction_IsentropicVortex<dim,nspecies,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::kelvin_helmholtz_instability) {
        if constexpr (dim>1 && nstate==dim+2) return std::make_shared<InitialConditionFunction_KHI<dim,nspecies,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::non_periodic_cube_flow) {
        if constexpr (dim==2 && nstate==1)  return std::make_shared<InitialConditionFunction_Zero<dim,nspecies,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::sod_shock_tube) {
        if constexpr (dim == 1 && nstate == dim+2)  return std::make_shared<InitialConditionFunction_SodShockTube<dim,nspecies,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::low_density) {
        if constexpr (dim < 3 && nstate == dim+2)  return std::make_shared<InitialConditionFunction_LowDensity<dim,nspecies,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::leblanc_shock_tube) {
        if constexpr (dim == 1 && nstate == dim+2)  return std::make_shared<InitialConditionFunction_LeblancShockTube<dim,nspecies,nstate,real> > (param);
    } else if (flow_type == FlowCaseEnum::shu_osher_problem) {
        if constexpr (dim == 1 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_ShuOsherProblem<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::double_mach_reflection) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_DoubleMachReflection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::shock_diffraction) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_ShockDiffraction<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::astrophysical_jet) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_AstrophysicalJet<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::strong_vortex_shock_wave) {
        if constexpr (dim == 2 && nstate == dim + 2)  return std::make_shared<InitialConditionFunction_SVSW<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::advection_limiter) {
        if constexpr (dim < 3 && nstate == 1)  return std::make_shared<InitialConditionFunction_Advection<dim,nspecies,nstate,real> >();
    } else if (flow_type == FlowCaseEnum::burgers_limiter) {
        if constexpr (nstate==dim && dim<3) return std::make_shared<InitialConditionFunction_BurgersInviscid<dim,nspecies,nstate,real> >();
    } else if (flow_type == FlowCaseEnum::multi_species_acoustic_wave) {
        if constexpr (dim==2 && nspecies==1 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_AcousticWave_MultiSpecies<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_vortex_advection) {
        if constexpr (dim==1 && (nspecies==2||nspecies==3) && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_VortexAdvection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_high_temperature_vortex_advection) {
        if constexpr (dim==1 && (nspecies==2||nspecies==3) && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection) {
        if constexpr (dim==1 && (nspecies==2||nspecies==3) && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_isentropic_euler_vortex) {
        if constexpr (dim==2 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_IsentropicEulerVortex<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_two_dimensional_vortex_advection) {
        if constexpr (dim==2 && nspecies==2 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_fuel_drop_advection) {
        if constexpr (dim==2 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_FuelDropAdvection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_three_dimensional_vortex_advection) {
        if constexpr (dim==3 && nspecies==2 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_taylor_green_vortex) {
        if constexpr (dim==3 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_TaylorGreenVortex<dim,nspecies,nstate,real> >(param);
    } else if (flow_type == FlowCaseEnum::multi_species_mixture_taylor_green_vortex) {
        if constexpr (dim==3 && nstate==dim+2+nspecies-1) return std::make_shared<InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex<dim,nspecies,nstate,real> >(param);
    } else {
        std::cout << "Invalid Flow Case Type. You probably forgot to add it to the list of flow cases in initial_condition_function.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

#if PHILIP_SPECIES==1
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, 1, double>;
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, 2, double>;
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, 3, double>;
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, 4, double>;
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, 5, double>;
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, 6, double>;

template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES, 1, double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES, 2, double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES, 3, double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES, 4, double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES, 5, double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES, 6, double>;

// functions instantiated for all dim
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, 1, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, 2, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, 3, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, 4, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, 5, double>;
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, 6, double>;
template class InitialConditionFunction_Advection <PHILIP_DIM, 1, 1, double>;
template class InitialConditionFunction_BurgersInviscid <PHILIP_DIM, 1, PHILIP_DIM, double>;
template class InitialConditionFunction_AdvectionEnergy <PHILIP_DIM, 1, 1, double>;
template class InitialConditionFunction_ConvDiff <PHILIP_DIM, 1, 1, double>;
template class InitialConditionFunction_ConvDiffEnergy <PHILIP_DIM, 1, 1,double>;


#if PHILIP_DIM==1
template class InitialConditionFunction_BurgersViscous <PHILIP_DIM, PHILIP_SPECIES, 1, double>;
template class InitialConditionFunction_BurgersRewienski <PHILIP_DIM, PHILIP_SPECIES, 1, double>;
template class InitialConditionFunction_BurgersInviscidEnergy <PHILIP_DIM, PHILIP_SPECIES, 1, double>;
template class InitialConditionFunction_EulerBase <PHILIP_DIM,PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_SodShockTube <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_LeblancShockTube <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_ShuOsherProblem <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM + 2, double>;
#endif

#if PHILIP_DIM==3
template class InitialConditionFunction_TaylorGreenVortex <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_TaylorGreenVortex_Isothermal <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM>1
template class InitialConditionFunction_IsentropicVortex <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM==2
template class InitialConditionFunction_KHI <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_EulerBase <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM + 2, double>;
template class InitialConditionFunction_DoubleMachReflection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_ShockDiffraction <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_AstrophysicalJet <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
template class InitialConditionFunction_SVSW <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
#endif

#if PHILIP_DIM < 3
template class InitialConditionFunction_LowDensity <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
#endif

#else
template class InitialConditionFunction <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+(PHILIP_SPECIES-1), double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_SPECIES,  PHILIP_DIM+2+(PHILIP_SPECIES-1), double>;
template class InitialConditionFunction_Zero <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+(PHILIP_SPECIES-1), double>;

    #if PHILIP_DIM==3 && PHILIP_SPECIES==2
    template class InitialConditionFunction_MultiSpecies_ThreeDimensional_VortexAdvection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    template class InitialConditionFunction_MultiSpecies_TaylorGreenVortex <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    template class InitialConditionFunction_MultiSpecies_Mixture_TaylorGreenVortex <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    #endif

    #if PHILIP_DIM==2 && PHILIP_SPECIES==2
    template class InitialConditionFunction_MultiSpecies_IsentropicEulerVortex <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    template class InitialConditionFunction_MultiSpecies_TwoDimensional_VortexAdvection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    template class InitialConditionFunction_MultiSpecies_FuelDropAdvection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    #endif

    #if PHILIP_DIM==1 && PHILIP_SPECIES > 1 && PHILIP_SPECIES < 4
    template class InitialConditionFunction_MultiSpecies_HighTemperature_VortexAdvection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    template class InitialConditionFunction_MultiSpecies_VortexAdvection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    template class InitialConditionFunction_MultiSpecies_CaloricallyPerfect_Euler_VortexAdvection <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double>;
    #endif

#endif

} // PHiLiP namespace
