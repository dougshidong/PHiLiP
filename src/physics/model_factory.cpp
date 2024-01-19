#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

#include <deal.II/base/tensor.h>

#include "ADTypes.hpp"

#include "model_factory.h"
#include "manufactured_solution.h"
#include "large_eddy_simulation.h"
#include "reynolds_averaged_navier_stokes.h"
#include "negative_spalart_allmaras_rans_model.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::shared_ptr < ModelBase<dim,nstate,real> >
ModelFactory<dim,nstate,real>
::create_Model(const Parameters::AllParameters *const parameters_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PDE_enum pde_type = parameters_input->pde_type;

    if(pde_type == PDE_enum::physics_model) {
        // generating the manufactured solution from the manufactured solution factory
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> >  manufactured_solution_function 
            = ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(parameters_input, nstate);

        using Model_enum = Parameters::AllParameters::ModelType;
        Model_enum model_type = parameters_input->model_type;
        
        // ===============================================================================
        // Model
        // ===============================================================================
        // -------------------------------------------------------------------------------
        // Large Eddy Simulation (LES)
        // -------------------------------------------------------------------------------
        if (model_type == Model_enum::large_eddy_simulation) {
            if constexpr ((nstate==dim+2) && (dim==3)) {
                // Create Large Eddy Simulation (LES) model based on the SGS model type
                using SGS_enum = Parameters::PhysicsModelParam::SubGridScaleModel;
                SGS_enum sgs_model_type = parameters_input->physics_model_param.SGS_model_type;
                if (sgs_model_type == SGS_enum::smagorinsky) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Smagorinsky model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_Smagorinsky<dim,nstate,real> > (
                        parameters_input,
                        parameters_input->euler_param.ref_length,
                        parameters_input->euler_param.gamma_gas,
                        parameters_input->euler_param.mach_inf,
                        parameters_input->euler_param.angle_of_attack,
                        parameters_input->euler_param.side_slip_angle,
                        parameters_input->navier_stokes_param.prandtl_number,
                        parameters_input->navier_stokes_param.reynolds_number_inf,
                        parameters_input->navier_stokes_param.use_constant_viscosity,
                        parameters_input->navier_stokes_param.nondimensionalized_constant_viscosity,
                        parameters_input->navier_stokes_param.temperature_inf,
                        parameters_input->physics_model_param.turbulent_prandtl_number,
                        parameters_input->physics_model_param.ratio_of_filter_width_to_cell_size,
                        parameters_input->physics_model_param.smagorinsky_model_constant,
                        parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                        parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                        manufactured_solution_function,
                        parameters_input->two_point_num_flux_type);
                } else if (sgs_model_type == SGS_enum::wall_adaptive_local_eddy_viscosity) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_WALE<dim,nstate,real> > (
                        parameters_input,
                        parameters_input->euler_param.ref_length,
                        parameters_input->euler_param.gamma_gas,
                        parameters_input->euler_param.mach_inf,
                        parameters_input->euler_param.angle_of_attack,
                        parameters_input->euler_param.side_slip_angle,
                        parameters_input->navier_stokes_param.prandtl_number,
                        parameters_input->navier_stokes_param.reynolds_number_inf,
                        parameters_input->navier_stokes_param.use_constant_viscosity,
                        parameters_input->navier_stokes_param.nondimensionalized_constant_viscosity,
                        parameters_input->navier_stokes_param.temperature_inf,
                        parameters_input->physics_model_param.turbulent_prandtl_number,
                        parameters_input->physics_model_param.ratio_of_filter_width_to_cell_size,
                        parameters_input->physics_model_param.WALE_model_constant,
                        parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                        parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                        manufactured_solution_function,
                        parameters_input->two_point_num_flux_type);
                } else if (sgs_model_type == SGS_enum::vreman) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Vreman eddy viscosity model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_Vreman<dim,nstate,real> > (
                        parameters_input,
                        parameters_input->euler_param.ref_length,
                        parameters_input->euler_param.gamma_gas,
                        parameters_input->euler_param.mach_inf,
                        parameters_input->euler_param.angle_of_attack,
                        parameters_input->euler_param.side_slip_angle,
                        parameters_input->navier_stokes_param.prandtl_number,
                        parameters_input->navier_stokes_param.reynolds_number_inf,
                        parameters_input->navier_stokes_param.use_constant_viscosity,
                        parameters_input->navier_stokes_param.nondimensionalized_constant_viscosity,
                        parameters_input->navier_stokes_param.temperature_inf,
                        parameters_input->physics_model_param.turbulent_prandtl_number,
                        parameters_input->physics_model_param.ratio_of_filter_width_to_cell_size,
                        parameters_input->physics_model_param.vreman_model_constant,
                        parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                        parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                        manufactured_solution_function,
                        parameters_input->two_point_num_flux_type);
                } 
                else {
                    std::cout << "Can't create LargeEddySimulationBase, invalid SGSModelType type: " << sgs_model_type << std::endl;
                    assert(0==1 && "Can't create LargeEddySimulationBase, invalid SGSModelType type");
                    return nullptr;
                }
            } 
            else {
                // LES does not exist for nstate!=(dim+2) || dim!=3
                manufactured_solution_function = nullptr;
                return nullptr;
            }
        } 
        // -------------------------------------------------------------------------------
        // Reynolds-Averaged Navier-Stokes (RANS) + RANS model
        // -------------------------------------------------------------------------------
        else if (model_type == Model_enum::reynolds_averaged_navier_stokes) {
            using RANSModel_enum = Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel;
            RANSModel_enum rans_model_type = parameters_input->physics_model_param.RANS_model_type;  
            // Create Reynolds-Averaged Navier-Stokes (RANS) model with one-equation model  
            if(rans_model_type == RANSModel_enum::SA_negative){
                if constexpr (nstate==dim+3) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // SA negative model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -      
                    return std::make_shared < ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real> > (
                        parameters_input,
                        parameters_input->euler_param.ref_length,
                        parameters_input->euler_param.gamma_gas,
                        parameters_input->euler_param.mach_inf,
                        parameters_input->euler_param.angle_of_attack,
                        parameters_input->euler_param.side_slip_angle,
                        parameters_input->navier_stokes_param.prandtl_number,
                        parameters_input->navier_stokes_param.reynolds_number_inf,
                        parameters_input->navier_stokes_param.use_constant_viscosity,
                        parameters_input->navier_stokes_param.nondimensionalized_constant_viscosity,
                        parameters_input->physics_model_param.turbulent_prandtl_number,
                        parameters_input->navier_stokes_param.temperature_inf,
                        parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                        parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                        manufactured_solution_function,
                        parameters_input->two_point_num_flux_type);
                }
                else {
                    // SA negative does not exist for nstate!=(dim+3)
                    manufactured_solution_function = nullptr;
                    return nullptr;
                }   
            }
            else {
                    std::cout << "Can't create ReynoldsAveragedNavierStokesBase, invalid RANSModelType type: " << rans_model_type << std::endl;
                    assert(0==1 && "Can't create ReynoldsAveragedNavierStokesBase, invalid RANSModelType type");
                    return nullptr;
            }
        }
        else {
            // prevent warnings for dim=3,nstate=4, etc.
            // to avoid "unused variable" warnings
            std::cout << "Can't create ModelBase, invalid ModelType type: " << model_type << std::endl;
            assert(0==1 && "Can't create ModelBase, invalid ModelType type");
            manufactured_solution_function = nullptr;
            return nullptr;
        }
    } 
    else {
        return nullptr;
    }
}

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
/// TO DO: adding tempalte from 1 to "maximum n_state" = "5+maximum_n_species" 1 by 1, boost loop
// Instantiate explicitly
template class ModelFactory<PHILIP_DIM, 1, double>;
template class ModelFactory<PHILIP_DIM, 2, double>;
template class ModelFactory<PHILIP_DIM, 3, double>;
template class ModelFactory<PHILIP_DIM, 4, double>;
template class ModelFactory<PHILIP_DIM, 5, double>;
template class ModelFactory<PHILIP_DIM, 6, double>;
template class ModelFactory<PHILIP_DIM, 7, double>;
template class ModelFactory<PHILIP_DIM, 8, double>;
template class ModelFactory<PHILIP_DIM, 9, double>;
template class ModelFactory<PHILIP_DIM, 10, double>;
template class ModelFactory<PHILIP_DIM, 11, double>;
template class ModelFactory<PHILIP_DIM, 12, double>;
template class ModelFactory<PHILIP_DIM, 13, double>;
template class ModelFactory<PHILIP_DIM, 14, double>;
template class ModelFactory<PHILIP_DIM, 15, double>;
template class ModelFactory<PHILIP_DIM, 5+11, double>; // "maximum n_state" = "5+maximum_n_species"

template class ModelFactory<PHILIP_DIM, 1, FadType>;
template class ModelFactory<PHILIP_DIM, 2, FadType>;
template class ModelFactory<PHILIP_DIM, 3, FadType>;
template class ModelFactory<PHILIP_DIM, 4, FadType>;
template class ModelFactory<PHILIP_DIM, 5, FadType>;
template class ModelFactory<PHILIP_DIM, 6, FadType>;
template class ModelFactory<PHILIP_DIM, 7, FadType>;
template class ModelFactory<PHILIP_DIM, 8, FadType>;
template class ModelFactory<PHILIP_DIM, 9, FadType>;
template class ModelFactory<PHILIP_DIM, 10, FadType>;
template class ModelFactory<PHILIP_DIM, 11, FadType>;
template class ModelFactory<PHILIP_DIM, 12, FadType>;
template class ModelFactory<PHILIP_DIM, 13, FadType>;
template class ModelFactory<PHILIP_DIM, 14, FadType>;
template class ModelFactory<PHILIP_DIM, 15, FadType>;
template class ModelFactory<PHILIP_DIM, 5+11, FadType>; // "maximum n_state" = "5+maximum_n_species"

template class ModelFactory<PHILIP_DIM, 1, RadType>;
template class ModelFactory<PHILIP_DIM, 2, RadType>;
template class ModelFactory<PHILIP_DIM, 3, RadType>;
template class ModelFactory<PHILIP_DIM, 4, RadType>;
template class ModelFactory<PHILIP_DIM, 5, RadType>;
template class ModelFactory<PHILIP_DIM, 6, RadType>;
template class ModelFactory<PHILIP_DIM, 7, RadType>;
template class ModelFactory<PHILIP_DIM, 8, RadType>;
template class ModelFactory<PHILIP_DIM, 9, RadType>;
template class ModelFactory<PHILIP_DIM, 10, RadType>;
template class ModelFactory<PHILIP_DIM, 11, RadType>;
template class ModelFactory<PHILIP_DIM, 12, RadType>;
template class ModelFactory<PHILIP_DIM, 13, RadType>;
template class ModelFactory<PHILIP_DIM, 14, RadType>;
template class ModelFactory<PHILIP_DIM, 15, RadType>;
template class ModelFactory<PHILIP_DIM, 5+11, RadType>; // "maximum n_state" = "5+maximum_n_species"

template class ModelFactory<PHILIP_DIM, 1, FadFadType>;
template class ModelFactory<PHILIP_DIM, 2, FadFadType>;
template class ModelFactory<PHILIP_DIM, 3, FadFadType>;
template class ModelFactory<PHILIP_DIM, 4, FadFadType>;
template class ModelFactory<PHILIP_DIM, 5, FadFadType>;
template class ModelFactory<PHILIP_DIM, 6, FadFadType>;
template class ModelFactory<PHILIP_DIM, 7, FadFadType>;
template class ModelFactory<PHILIP_DIM, 8, FadFadType>;
template class ModelFactory<PHILIP_DIM, 9, FadFadType>;
template class ModelFactory<PHILIP_DIM, 10, FadFadType>;
template class ModelFactory<PHILIP_DIM, 11, FadFadType>;
template class ModelFactory<PHILIP_DIM, 12, FadFadType>;
template class ModelFactory<PHILIP_DIM, 13, FadFadType>;
template class ModelFactory<PHILIP_DIM, 14, FadFadType>;
template class ModelFactory<PHILIP_DIM, 15, FadFadType>;
template class ModelFactory<PHILIP_DIM, 5+11, FadFadType>; // "maximum n_state" = "5+maximum_n_species"

template class ModelFactory<PHILIP_DIM, 1, RadFadType>;
template class ModelFactory<PHILIP_DIM, 2, RadFadType>;
template class ModelFactory<PHILIP_DIM, 3, RadFadType>;
template class ModelFactory<PHILIP_DIM, 4, RadFadType>;
template class ModelFactory<PHILIP_DIM, 5, RadFadType>;
template class ModelFactory<PHILIP_DIM, 6, RadFadType>;
template class ModelFactory<PHILIP_DIM, 7, RadFadType>;
template class ModelFactory<PHILIP_DIM, 8, RadFadType>;
template class ModelFactory<PHILIP_DIM, 9, RadFadType>;
template class ModelFactory<PHILIP_DIM, 10, RadFadType>;
template class ModelFactory<PHILIP_DIM, 11, RadFadType>;
template class ModelFactory<PHILIP_DIM, 12, RadFadType>;
template class ModelFactory<PHILIP_DIM, 13, RadFadType>;
template class ModelFactory<PHILIP_DIM, 14, RadFadType>;
template class ModelFactory<PHILIP_DIM, 15, RadFadType>;
template class ModelFactory<PHILIP_DIM, 5+11, RadFadType>; // "maximum n_state" = "5+maximum_n_species"

} // Physics namespace
} // PHiLiP namespace

