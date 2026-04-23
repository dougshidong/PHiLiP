#include <boost/preprocessor/seq/for_each.hpp>

#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

#include <deal.II/base/tensor.h>

#include "ADTypes.hpp"

#include "model_factory.h"
#include "manufactured_solution.h"
#include "large_eddy_simulation.h"
#include "reynolds_averaged_navier_stokes.h"
#include "negative_spalart_allmaras_rans_model.h"
#include "navier_stokes_model.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nspecies, int nstate, typename real>
std::shared_ptr < ModelBase<dim,nspecies,nstate,real> >
ModelFactory<dim,nspecies,nstate,real>
::create_Model(const Parameters::AllParameters *const parameters_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PDE_enum pde_type = parameters_input->pde_type;

    if((pde_type == PDE_enum::physics_model || pde_type == PDE_enum::physics_model_filtered) && nspecies==1) {
        // generating the manufactured solution from the manufactured solution factory
        std::shared_ptr< ManufacturedSolutionFunction<dim,nspecies,real> >  manufactured_solution_function 
            = ManufacturedSolutionFactory<dim,nspecies,real>::create_ManufacturedSolution(parameters_input, nstate);

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
                    return std::make_shared < LargeEddySimulation_Smagorinsky<dim,nspecies,nstate,real> > (
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
                        parameters_input->two_point_num_flux_type,
                        parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                } else if (sgs_model_type == SGS_enum::wall_adaptive_local_eddy_viscosity) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_WALE<dim,nspecies,nstate,real> > (
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
                        parameters_input->two_point_num_flux_type,
                        parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                } else if (sgs_model_type == SGS_enum::vreman) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Vreman eddy viscosity model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_Vreman<dim,nspecies,nstate,real> > (
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
                        parameters_input->two_point_num_flux_type,
                        parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                } else if (sgs_model_type == SGS_enum::shear_improved_smagorinsky) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Shear-improved Smagorinsky model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_ShearImprovedSmagorinsky<dim,nspecies,nstate,real> > (
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
                        parameters_input->two_point_num_flux_type,
                        parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                } else if ((sgs_model_type == SGS_enum::small_small_variational_multiscale) ||
                           (sgs_model_type == SGS_enum::all_all_variational_multiscale)) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Variational multiscale (VMS) eddy viscosity models
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    const double domain_left = parameters_input->flow_solver_param.grid_left_bound;
                    const double domain_right = parameters_input->flow_solver_param.grid_right_bound;
                    const int number_of_cells_per_direction = parameters_input->flow_solver_param.number_of_grid_elements_per_dimension;
                    const double mesh_size = (domain_right - domain_left)/((double)number_of_cells_per_direction);

                    if (sgs_model_type == SGS_enum::small_small_variational_multiscale) {
                        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        // Small-Small Variational multiscale (SmallSmallVMS) eddy viscosity model
                        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        return std::make_shared < LargeEddySimulation_SmallSmallVMS<dim,nspecies,nstate,real> > (
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
                            parameters_input->flow_solver_param.poly_degree,
                            parameters_input->physics_model_param.poly_degree_max_large_scales,
                            mesh_size,
                            parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                            parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                            manufactured_solution_function,
                            parameters_input->two_point_num_flux_type,
                            parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                    } else if (sgs_model_type == SGS_enum::all_all_variational_multiscale) {
                        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        // All-All Variational multiscale (AllAllVMS) eddy viscosity model
                        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        return std::make_shared < LargeEddySimulation_AllAllVMS<dim,nspecies,nstate,real> > (
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
                            parameters_input->flow_solver_param.poly_degree,
                            parameters_input->physics_model_param.poly_degree_max_large_scales,
                            mesh_size,
                            parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                            parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                            manufactured_solution_function,
                            parameters_input->two_point_num_flux_type,
                            parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                    } else {
                        std::cout << "Can't create LargeEddySimulationVMS, invalid SGSModelType type: " << sgs_model_type << std::endl;
                        assert(0==1 && "Can't create LargeEddySimulationVMS, invalid SGSModelType type");
                        return nullptr;
                    }
                } else if (sgs_model_type == SGS_enum::dynamic_smagorinsky) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Dynamic Smagorinsky model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_DynamicSmagorinsky<dim,nspecies,nstate,real> > (
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
                        parameters_input->two_point_num_flux_type,
                        parameters_input->physics_model_param.apply_low_reynolds_number_eddy_viscosity_correction);
                } 
                else {
                    std::cout << "Can't create LargeEddySimulationBase, invalid SGSModelType type: " << sgs_model_type << std::endl;
                    assert(0==1 && "Can't create LargeEddySimulationBase, invalid SGSModelType type");
                    return nullptr;
                }
            } 
            else {
                // LES does not exist for nstate!=(dim+2) || dim!=3
                std::cout << "Can't create LES for nstate!=(dim+2) or dim!=3" << std::endl;
                assert(0==1 && "Can't create LES for nstate!=(dim+2) or dim!=3");
                manufactured_solution_function = nullptr;
                return nullptr;
            }
        }
        // -------------------------------------------------------------------------------
        // Navier-Stokes model
        // -------------------------------------------------------------------------------
        else if (model_type == Model_enum::navier_stokes_model) {
            if constexpr ((nstate==dim+2) && (dim==3)) {
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                // Navier-Stokes with model source terms (e.g. channel flow)
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                return std::make_shared < NavierStokesWithModelSourceTerms<dim,nspecies,nstate,real> > (
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
                    parameters_input->flow_solver_param.relaxation_coefficient_for_turbulent_channel_flow_source_term,
                    parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
                    parameters_input->navier_stokes_param.thermal_boundary_condition_type,
                    manufactured_solution_function,
                    parameters_input->two_point_num_flux_type);
            } else {
                // Navier-Stokes model does not exist for nstate!=(dim+2) || dim!=3
                std::cout << "Can't create Navier-Stokes model for nstate!=(dim+2) or dim!=3" << std::endl;
                assert(0==1 && "Can't create Navier-Stokes model for nstate!=(dim+2) or dim!=3");
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
                    return std::make_shared < ReynoldsAveragedNavierStokes_SAneg<dim,nspecies,nstate,real> > (
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
                    std::cout << "Can't create RANS for nstate!=(dim+2) or dim!=3" << std::endl;
                    assert(0==1 && "Can't create RANS for nstate!=(dim+2) or dim!=3");
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
// Instantiate explicitly
#if PHILIP_SPECIES==1
    // Define a sequence of indices representing the range of nstate
    #define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)(8)

    // Define a macro to instantiate functions for a specific nstate
    #define INSTANTIATE_FOR_NSTATE(r, data, nstate) \
        template class ModelFactory<PHILIP_DIM, PHILIP_SPECIES, nstate, double>; \
        template class ModelFactory<PHILIP_DIM, PHILIP_SPECIES, nstate, FadType>; \
        template class ModelFactory<PHILIP_DIM, PHILIP_SPECIES, nstate, RadType>; \
        template class ModelFactory<PHILIP_DIM, PHILIP_SPECIES, nstate, FadFadType>; \
        template class ModelFactory<PHILIP_DIM, PHILIP_SPECIES, nstate, RadFadType>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FOR_NSTATE, _, POSSIBLE_NSTATE)
#else
    #define POSSIBLE_TYPE (double)(FadType)(RadType)(FadFadType)(RadFadType)
    #define INSTANTIATE_TYPES(r, data, type) \
        template class ModelFactory<PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, type>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TYPES, _, POSSIBLE_TYPE)
#endif
} // Physics namespace
} // PHiLiP namespace

