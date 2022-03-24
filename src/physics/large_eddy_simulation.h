#ifndef __LARGE_EDDY_SIMULATION__
#define __LARGE_EDDY_SIMULATION__

#include "model.h"
#include "navier_stokes.h"
#include "euler.h"

namespace PHiLiP {
namespace Physics {

/// Large Eddy Simulation equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class LargeEddySimulationBase : public ModelBase <dim, nstate, real>
{
public:
    /// Constructor
	LargeEddySimulationBase(
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input,
	    const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              ratio_of_filter_width_to_cell_size);

    /// Destructor
    ~LargeEddySimulationBase() {};

    /// Turbulent Prandtl number
    const double turbulent_prandtl_number;

    /// Ratio of filter width to cell size
    const double ratio_of_filter_width_to_cell_size;

    /// Pointer to Navier-Stokes physics object
    std::unique_ptr< NavierStokes<dim,nstate,real> > navier_stokes_physics;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const dealii::types::global_dof_index cell_index) const;

    /// Compute the nondimensionalized filter width used by the SGS model given a cell index
    double get_filter_width (const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    virtual std::array<dealii::Tensor<1,dim,real>,dim> compute_SGS_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    virtual dealii::Tensor<1,dim,real> compute_SGS_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)* (Automatic Differentiation Type: FadType)
    virtual std::array<dealii::Tensor<1,dim,FadType>,dim> compute_SGS_stress_tensor_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<1,dim,FadType> compute_SGS_heat_flux_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

protected:
    /// Returns the square of the magnitude of the tensor
    template<typename real2> 
    real2 get_tensor_magnitude_sqr (const std::array<dealii::Tensor<1,dim,real2>,dim> &tensor) const;

    /// Templated dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /** Dissipative flux Jacobian (repeated from NavierStokes)
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const dealii::types::global_dof_index cell_index) const;

    /** Dissipative flux Jacobian wrt gradient component (repeated from NavierStokes)
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Get manufactured solution value (repeated from Euler)
    std::array<real,nstate> get_manufactured_solution_value (
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution value (repeated from Euler)
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient (
        const dealii::Point<dim,real> &pos) const;

    /// Dissipative flux contribution to the source term (repeated from NavierStokes)
    std::array<real,nstate> dissipative_source_term (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;
};

/// Smagorinsky eddy viscosity model. Derived from Large Eddy Simulation.
template <int dim, int nstate, typename real>
class LargeEddySimulation_Smagorinsky : public LargeEddySimulationBase <dim, nstate, real>
{
public:
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: To be put here
     */
    LargeEddySimulation_Smagorinsky(
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input,
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              ratio_of_filter_width_to_cell_size,
        const double                                              model_constant);

    /// SGS model constant
    const double model_constant;

    /// Destructor
    ~LargeEddySimulation_Smagorinsky() {};

    /// Returns the product of the eddy viscosity model constant and the filter width
    double get_model_constant_times_filter_width (const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    std::array<dealii::Tensor<1,dim,real>,dim> compute_SGS_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)* 
    dealii::Tensor<1,dim,real> compute_SGS_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)* (Automatic Differentiation Type: FadType)
    std::array<dealii::Tensor<1,dim,FadType>,dim> compute_SGS_stress_tensor_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)* (Automatic Differentiation Type: FadType)
    dealii::Tensor<1,dim,FadType> compute_SGS_heat_flux_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized eddy viscosity for the Smagorinsky model
    virtual real compute_eddy_viscosity(
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized eddy viscosity for the Smagorinsky model (Automatic Differentiation Type: FadType)
    virtual FadType compute_eddy_viscosity_fad(
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

protected:
    /// Templated nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    template<typename real2> std::array<dealii::Tensor<1,dim,real2>,dim> compute_SGS_stress_tensor_templated (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Templated nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    template<typename real2> 
    dealii::Tensor<1,dim,real2> compute_SGS_heat_flux_templated (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Templated scale nondimensionalized eddy viscosity for Smagorinsky model
    template<typename real2> real2 scale_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const real2 eddy_viscosity) const;

private:
    /// Templated nondimensionalized eddy viscosity for the Smagorinsky model.
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;
};

/// WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model. Derived from LargeEddySimulation_Smagorinsky for only modifying compute_eddy_viscosity.
template <int dim, int nstate, typename real>
class LargeEddySimulation_WALE : public LargeEddySimulation_Smagorinsky <dim, nstate, real>
{
public:
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    LargeEddySimulation_WALE(
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input,
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              ratio_of_filter_width_to_cell_size,
        const double                                              model_constant);

    /// Destructor
    ~LargeEddySimulation_WALE() {};

    /** Nondimensionalized eddy viscosity for the WALE model. 
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    real compute_eddy_viscosity(
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const override;

    /** Nondimensionalized eddy viscosity for the WALE model. (Automatic Differentiation Type: FadType)
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    FadType compute_eddy_viscosity_fad(
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const override;

private:
    /// Templated nondimensionalized eddy viscosity for the WALE model.
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif
