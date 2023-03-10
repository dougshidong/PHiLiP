#include <deal.II/base/function.h>
#include "initial_condition_function.h"

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
// Turbulent Channel Flow -- Initial Condition
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TurbulentChannelFlow<dim,nstate,real>
::InitialConditionFunction_TurbulentChannelFlow (
    const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_,
    const double channel_height_,
    const double channel_friction_velocity_reynolds_number_)
    : InitialConditionFunction<dim,nstate,real>()
    , navier_stokes_physics(navier_stokes_physics_)
    , channel_height(channel_height_)
    , half_channel_height(0.5*channel_height)
    , channel_friction_velocity_reynolds_number(channel_friction_velocity_reynolds_number_)
{}

template <int dim, int nstate, typename real>
inline real InitialConditionFunction_TurbulentChannelFlow<dim, nstate, real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    const real density = 1.0; // freestream non-dimensionalized -- TO DO: Confirm / update this
    
    /// Conservative solution
    std::array<real,nstate> primitive_soln;
    primitive_soln[0] = density;
    primitive_soln[2] = 0.0; // assuming y-velocity is zero
    primitive_soln[3] = 0.0; // assuming z-velocity is zero

    // for efficiency, we do not compute the Reichart law of the wall for states that do not require it
    if(istate!=1 || istate!=5) {
        return primitive_soln[istate]; // same as conservative_solution for these states
    } else {
        const real temperature = 1.0; // freestream non-dimensionalized -- TO DO: Confirm / update this

        // Get closest wall normal distance
        real y = point[1]; // represents distance normal to top/bottom wall (which ever is closer); y-domain bounds are [0, channel_height]
        // -- apply initial condition symmetrically w.r.t. the top/bottom walls of the channel
        if(y > half_channel_height){
            // top wall
            y = channel_height - y; // distance from wall
        }

        // Get friction velocity
        const real viscosity_coefficient = navier_stokes_physics.compute_viscosity_coefficient_from_temperature(temperature);
        const real friction_velocity = viscosity_coefficient*channel_friction_velocity_reynolds_number/(density*this->half_channel_height);

        // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
        // References: 
        /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
            "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

            (Original paper) J. M.  ̈Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
            on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4
            (2000).
        */
        const real kappa = 0.38; // von Karman's constant
        const real C = 4.1;
        const real y_plus = density*friction_velocity*y/viscosity_coefficient;
        const real u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0));
        const real x_velocity = u_plus*friction_velocity;
        
        // set x-velocity
        primitive_soln[1] = x_velocity;

        // set pressure
        primitive_soln[5] = navier_stokes_physics.compute_pressure_from_density_temperature(density, temperature);
        std::array<real,nstate> conservative_soln = navier_stokes_physics.convert_primitive_to_conservative(primitive_soln);

        return conservative_soln[istate];
    }
}
// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition (Uniform density)
// ========================================================
template <int dim, int nstate, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
::InitialConditionFunction_TaylorGreenVortex (
    const double       gamma_gas,
    const double       mach_inf)
    : InitialConditionFunction<dim,nstate,real>()
    , gamma_gas(gamma_gas)
    , mach_inf(mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{}

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
    if (dim == 3) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real w   = primitive_value(point,3);
        const real p   = primitive_value(point,4);

        // convert primitive to conservative solution
        if(istate==0) value = rho; // density
        if(istate==1) value = rho*u; // x-momentum
        if(istate==2) value = rho*v; // y-momentum
        if(istate==3) value = rho*w; // z-momentum
        if(istate==4) value = p/(this->gamma_gas-1.0) + 0.5*rho*(u*u + v*v + w*w); // total energy
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
    const double       gamma_gas,
    const double       mach_inf)
    : InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>(gamma_gas,mach_inf)
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
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf);
            } else if (density_initial_condition_type == DensityInitialConditionEnum::isothermal) {
                return std::make_shared<InitialConditionFunction_TaylorGreenVortex_Isothermal<dim,nstate,real> >(
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf);
            }
        }
    } else if (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence) {
        if constexpr (dim==3 && nstate==dim+2) return nullptr; // nullptr since DHIT case initializes values from file
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersRewienski<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot) {
        if constexpr (dim==1 && nstate==1) return std::make_shared<InitialConditionFunction_BurgersViscous<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::naca0012  || flow_type == FlowCaseEnum::gaussian_bump) {
        if constexpr (dim==2 && nstate==dim+2) {
            Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
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
    } else if (flow_type == FlowCaseEnum::sshock) {
        if constexpr (dim==2 && nstate==1)  return std::make_shared<InitialConditionFunction_Zero<dim,nstate,real> > ();
    } else if (flow_type == FlowCaseEnum::channel_flow) {
        if constexpr (dim==3 && nstate==dim+2) {
            Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_double = Physics::NavierStokes<dim, nstate, double>(
                    param->euler_param.ref_length,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf,
                    param->euler_param.angle_of_attack,
                    param->euler_param.side_slip_angle,
                    param->navier_stokes_param.prandtl_number,
                    param->navier_stokes_param.reynolds_number_inf,
                    param->navier_stokes_param.use_constant_viscosity,
                    param->navier_stokes_param.nondimensionalized_constant_viscosity,
                    param->navier_stokes_param.temperature_inf,
                    param->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                    param->navier_stokes_param.thermal_boundary_condition_type,
                    nullptr,
                    param->two_point_num_flux_type);
            return std::make_shared<InitialConditionFunction_TurbulentChannelFlow<dim,nstate,real>>(
                navier_stokes_physics_double,
                param->flow_solver_param.turbulent_channel_height,
                param->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number);
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
template class InitialConditionFunction_TurbulentChannelFlow <PHILIP_DIM, PHILIP_DIM+2, double>;
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
