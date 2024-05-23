#ifndef PHILIP_DG_BASE_STATE_HPP
#define PHILIP_DG_BASE_STATE_HPP

#include <deal.II/distributed/tria.h>

#include "dg_base.hpp"
#include "parameters/all_parameters.h"
namespace PHiLiP {

/// Abstract class templated on the number of state variables
/*  Contains the objects and functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGBaseState : public DGBase<dim, real, MeshType>
{
   protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGBase<dim,real,MeshType>::Triangulation;

   public:
    using DGBase<dim,real,MeshType>::all_parameters; ///< Input parameters.

    /// Constructor.
    DGBaseState(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// Contains the physics of the PDE with real type
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, real > > pde_physics_double;
    /// Contains the model terms of the PDEType == PhysicsModel with real type
    std::shared_ptr < Physics::ModelBase<dim, nstate, real > > pde_model_double;
    /// Convective numerical flux with real type
    std::unique_ptr < NumericalFlux::NumericalFluxConvective<dim, nstate, real > > conv_num_flux_double;
    /// Dissipative numerical flux with real type
    std::unique_ptr < NumericalFlux::NumericalFluxDissipative<dim, nstate, real > > diss_num_flux_double;
    /// Link to Artificial dissipation class (with three dissipation types, depending on the input).
    std::shared_ptr <ArtificialDissipationBase<dim,nstate>> artificial_dissip;

    /// Contains the physics of the PDE with FadType
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, FadType > > pde_physics_fad;
    /// Contains the model terms of the PDEType == PhysicsModel with FadType
    std::shared_ptr < Physics::ModelBase<dim, nstate, FadType > > pde_model_fad;
    /// Convective numerical flux with FadType
    std::unique_ptr < NumericalFlux::NumericalFluxConvective<dim, nstate, FadType > > conv_num_flux_fad;
    /// Dissipative numerical flux with FadType
    std::unique_ptr < NumericalFlux::NumericalFluxDissipative<dim, nstate, FadType > > diss_num_flux_fad;

    /// Contains the physics of the PDE with RadType
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, RadType > > pde_physics_rad;
    /// Contains the model terms of the PDEType == PhysicsModel with RadType
    std::shared_ptr < Physics::ModelBase<dim, nstate, RadType > > pde_model_rad;
    /// Convective numerical flux with RadType
    std::unique_ptr < NumericalFlux::NumericalFluxConvective<dim, nstate, RadType > > conv_num_flux_rad;
    /// Dissipative numerical flux with RadType
    std::unique_ptr < NumericalFlux::NumericalFluxDissipative<dim, nstate, RadType > > diss_num_flux_rad;

    /// Contains the physics of the PDE with FadFadType
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, FadFadType > > pde_physics_fad_fad;
    /// Contains the model terms of the PDEType == PhysicsModel with FadFadType
    std::shared_ptr < Physics::ModelBase<dim, nstate, FadFadType > > pde_model_fad_fad;
    /// Convective numerical flux with FadFadType
    std::unique_ptr < NumericalFlux::NumericalFluxConvective<dim, nstate, FadFadType > > conv_num_flux_fad_fad;
    /// Dissipative numerical flux with FadFadType
    std::unique_ptr < NumericalFlux::NumericalFluxDissipative<dim, nstate, FadFadType > > diss_num_flux_fad_fad;

    /// Contains the physics of the PDE with RadFadDtype
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, RadFadType > > pde_physics_rad_fad;
    /// Contains the model terms of the PDEType == PhysicsModel with RadFadType
    std::shared_ptr < Physics::ModelBase<dim, nstate, RadFadType > > pde_model_rad_fad;
    /// Convective numerical flux with RadFadDtype
    std::unique_ptr < NumericalFlux::NumericalFluxConvective<dim, nstate, RadFadType > > conv_num_flux_rad_fad;
    /// Dissipative numerical flux with RadFadDtype
    std::unique_ptr < NumericalFlux::NumericalFluxDissipative<dim, nstate, RadFadType > > diss_num_flux_rad_fad;

    /** Change the physics object.
     *  Must provide all the AD types to ensure that the derivatives are consistent.
     */
    void set_physics(
        std::shared_ptr< Physics::PhysicsBase<dim, nstate, real       > > pde_physics_double_input,
        std::shared_ptr< Physics::PhysicsBase<dim, nstate, FadType    > > pde_physics_fad_input,
        std::shared_ptr< Physics::PhysicsBase<dim, nstate, RadType    > > pde_physics_rad_input,
        std::shared_ptr< Physics::PhysicsBase<dim, nstate, FadFadType > > pde_physics_fad_fad_input,
        std::shared_ptr< Physics::PhysicsBase<dim, nstate, RadFadType > > pde_physics_rad_fad_input);

    /// Allocate the necessary variables declared in src/physics/model.h
    void allocate_model_variables();

    /// Update the necessary variables declared in src/physics/model.h
    void update_model_variables();

    /// Set use_auxiliary_eq flag
    void set_use_auxiliary_eq();

   protected:
    /// Evaluate the time it takes for the maximum wavespeed to cross the cell domain.
    /** Currently only uses the convective eigenvalues. Future changes would take in account
     *  the maximum diffusivity and take the minimum time between dx/conv_eig and dx*dx/max_visc
     *  to determine the minimum travel time of information.
     *
     *  Furthermore, a more robust implementation would convert the values to a Bezier basis where
     *  the maximum and minimum values would be bounded by the Bernstein modal coefficients.
     */
    real evaluate_CFL (std::vector< std::array<real,nstate> > soln_at_q, const real artificial_dissipation, const real cell_diameter, const unsigned int cell_degree);

    /// Reinitializes the numerical fluxes based on the current physics.
    /** Usually called after setting physics.
     */
    void reset_numerical_fluxes();

    const Physics::PhysicsBase<dim, nstate, double> & get_physics(double) const
    {
        return *pde_physics_double;
    } 
    const Physics::PhysicsBase<dim, nstate, FadType> & get_physics(FadType) const
    {
        return *pde_physics_fad;
    }
    const Physics::PhysicsBase<dim, nstate, RadType> & get_physics(RadType) const
    {
        return *pde_physics_rad;
    }
    const Physics::PhysicsBase<dim, nstate, FadFadType> & get_physics(FadFadType) const
    {
        return *pde_physics_fad_fad;
    }
    const Physics::PhysicsBase<dim, nstate, RadFadType> & get_physics(RadFadType) const
    {
        return *pde_physics_rad_fad;
    }


    const NumericalFlux::NumericalFluxConvective<dim, nstate, double> & get_conv_num_flux(double) const
    {
        return *conv_num_flux_double;
    } 
    const NumericalFlux::NumericalFluxConvective<dim, nstate, FadType> & get_conv_num_flux(FadType) const
    {
        return *conv_num_flux_fad;
    }
    const NumericalFlux::NumericalFluxConvective<dim, nstate, RadType> & get_conv_num_flux(RadType) const
    {
        return *conv_num_flux_rad;
    }
    const NumericalFlux::NumericalFluxConvective<dim, nstate, FadFadType> & get_conv_num_flux(FadFadType) const
    {
        return *conv_num_flux_fad_fad;
    }
    const NumericalFlux::NumericalFluxConvective<dim, nstate, RadFadType> & get_conv_num_flux(RadFadType) const
    {
        return *conv_num_flux_rad_fad;
    }


    const NumericalFlux::NumericalFluxDissipative<dim, nstate, double> & get_diss_num_flux(double) const
    {
        return *diss_num_flux_double;
    } 
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, FadType> & get_diss_num_flux(FadType) const
    {
        return *diss_num_flux_fad;
    }
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, RadType> & get_diss_num_flux(RadType) const
    {
        return *diss_num_flux_rad;
    }
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, FadFadType> & get_diss_num_flux(FadFadType) const
    {
        return *diss_num_flux_fad_fad;
    }
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, RadFadType> & get_diss_num_flux(RadFadType) const
    {
        return *diss_num_flux_rad_fad;
    }
}; // end of DGBaseState class

}  // namespace PHiLiP

#endif  // PHILIP_DG_BASE_STATE_HPP
