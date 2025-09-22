#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
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

// Instantiate explicitly

// Define a sequence of indices representing the range [1, 8]
#define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)(7)(8)

// Define a macro to instantiate MyTemplate for a specific index
#define INSTANTIATE_DOUBLE(r, data, index) \
    template class ModelFactory <PHILIP_DIM, index, double>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_DOUBLE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_FADTYPE(r, data, index) \
    template class ModelFactory <PHILIP_DIM, index, FadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FADTYPE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_RADTYPE(r, data, index) \
    template class ModelFactory <PHILIP_DIM, index, RadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_RADTYPE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_FADFADTYPE(r, data, index) \
    template class ModelFactory <PHILIP_DIM, index, FadFadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FADFADTYPE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_RADFADTYPE(r, data, index) \
    template class ModelFactory <PHILIP_DIM, index, RadFadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_RADFADTYPE, _, POSSIBLE_NSTATE)

// Templated to allow compilation when NUMBER_OF_SPECIES > 2, but may not work.
#if ((PHILIP_DIM+2+(PHILIP_SPECIES-1)) > 8)
    #define INSTANTIATE_DOUBLE(r, data, index) \
        template class ModelFactory <PHILIP_DIM, index, double>;
    BOOST_PP_REPEAT_FROM_TO_d(0,PHILIP_DIM+2+(PHILIP_SPECIES-1),INSTANTIATE_DOUBLE, _)

    #define INSTANTIATE_FADTYPE(r, data, index) \
        template class ModelFactory <PHILIP_DIM, index, FadType>;
    BOOST_PP_REPEAT_FROM_TO_d(0,PHILIP_DIM+2+(PHILIP_SPECIES-1),INSTANTIATE_FADTYPE, _)

    #define INSTANTIATE_RADTYPE(r, data, index) \
        template class ModelFactory <PHILIP_DIM, index, RadType>;
    BOOST_PP_REPEAT_FROM_TO_d(0,PHILIP_DIM+2+(PHILIP_SPECIES-1),INSTANTIATE_RADTYPE, _)

    #define INSTANTIATE_FADFADTYPE(r, data, index) \
        template class ModelFactory <PHILIP_DIM, index, FadFadType>;
    BOOST_PP_REPEAT_FROM_TO_d(0,PHILIP_DIM+2+(PHILIP_SPECIES-1),INSTANTIATE_FADFADTYPE, _)

    #define INSTANTIATE_RADFADTYPE(r, data, index) \
        template class ModelFactory <PHILIP_DIM, index, RadFadType>;
    BOOST_PP_REPEAT_FROM_TO_d(0,PHILIP_DIM+2+(PHILIP_SPECIES-1),INSTANTIATE_RADFADTYPE, _)
#endif
} // Physics namespace
} // PHiLiP namespace

