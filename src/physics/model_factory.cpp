#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

#include <deal.II/base/tensor.h>

#include "ADTypes.hpp"

#include "model_factory.h"
#include "manufactured_solution.h"
#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::shared_ptr < ModelBase<dim,nstate,real> >
ModelFactory<dim,nstate,real>
::create_Model(const Parameters::AllParameters *const parameters_input,
               const double                    filter_width)
{
    // In the future, we could create an object / class for holding/updating inputs such as grid spacing etc and just pass a pointer to the object
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
            if constexpr (nstate==dim+2) {
                // Create Large Eddy Simulation (LES) model based on the SGS model type
                using SGS_enum = Parameters::PhysicsModelParam::SubGridScaleModel;
                SGS_enum sgs_model_type = parameters_input->physics_model_param.SGS_model_type;
                if (sgs_model_type == SGS_enum::smagorinsky) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // Smagorinsky model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_Smagorinsky<dim,nstate,real> > (
                        manufactured_solution_function,
                        parameters_input->euler_param.ref_length,
                        parameters_input->euler_param.gamma_gas,
                        parameters_input->euler_param.mach_inf,
                        parameters_input->euler_param.angle_of_attack,
                        parameters_input->euler_param.side_slip_angle,
                        parameters_input->navier_stokes_param.prandtl_number,
                        parameters_input->navier_stokes_param.reynolds_number_inf,
                        parameters_input->physics_model_param.turbulent_prandtl_number,
                        parameters_input->physics_model_param.smagorinsky_model_constant,
                        filter_width);
                } else if (sgs_model_type == SGS_enum::wall_adaptive_local_eddy_viscosity) {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
                    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    return std::make_shared < LargeEddySimulation_WALE<dim,nstate,real> > (
                        manufactured_solution_function,
                        parameters_input->euler_param.ref_length,
                        parameters_input->euler_param.gamma_gas,
                        parameters_input->euler_param.mach_inf,
                        parameters_input->euler_param.angle_of_attack,
                        parameters_input->euler_param.side_slip_angle,
                        parameters_input->navier_stokes_param.prandtl_number,
                        parameters_input->navier_stokes_param.reynolds_number_inf,
                        parameters_input->physics_model_param.turbulent_prandtl_number,
                        parameters_input->physics_model_param.WALE_model_constant,
                        filter_width);
                } else {
                    std::cout << "Can't create LargeEddySimulationBase, invalid SGSModelType type: " << sgs_model_type << std::endl;
                    assert(0==1 && "Can't create LargeEddySimulationBase, invalid SGSModelType type");
                    return nullptr;
                }
            }
        } else {
            // prevent warnings for dim=3,nstate=4, etc.
            // to avoid "unused variable" warnings
            manufactured_solution_function = nullptr;
        }    
        std::cout << "Can't create ModelBase, invalid ModelType type: " << model_type << std::endl;
        assert(0==1 && "Can't create ModelBase, invalid ModelType type");
        return nullptr;
    } else {
        // if pde_type != PhysicsModel
        (void) filter_width;
        return nullptr;
    }
}

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class ModelFactory<PHILIP_DIM, 1, double>;
template class ModelFactory<PHILIP_DIM, 2, double>;
template class ModelFactory<PHILIP_DIM, 3, double>;
template class ModelFactory<PHILIP_DIM, 4, double>;
template class ModelFactory<PHILIP_DIM, 5, double>;
template class ModelFactory<PHILIP_DIM, 8, double>;

template class ModelFactory<PHILIP_DIM, 1, FadType>;
template class ModelFactory<PHILIP_DIM, 2, FadType>;
template class ModelFactory<PHILIP_DIM, 3, FadType>;
template class ModelFactory<PHILIP_DIM, 4, FadType>;
template class ModelFactory<PHILIP_DIM, 5, FadType>;
template class ModelFactory<PHILIP_DIM, 8, FadType>;

template class ModelFactory<PHILIP_DIM, 1, RadType>;
template class ModelFactory<PHILIP_DIM, 2, RadType>;
template class ModelFactory<PHILIP_DIM, 3, RadType>;
template class ModelFactory<PHILIP_DIM, 4, RadType>;
template class ModelFactory<PHILIP_DIM, 5, RadType>;
template class ModelFactory<PHILIP_DIM, 8, RadType>;

template class ModelFactory<PHILIP_DIM, 1, FadFadType>;
template class ModelFactory<PHILIP_DIM, 2, FadFadType>;
template class ModelFactory<PHILIP_DIM, 3, FadFadType>;
template class ModelFactory<PHILIP_DIM, 4, FadFadType>;
template class ModelFactory<PHILIP_DIM, 5, FadFadType>;
template class ModelFactory<PHILIP_DIM, 8, FadFadType>;

template class ModelFactory<PHILIP_DIM, 1, RadFadType>;
template class ModelFactory<PHILIP_DIM, 2, RadFadType>;
template class ModelFactory<PHILIP_DIM, 3, RadFadType>;
template class ModelFactory<PHILIP_DIM, 4, RadFadType>;
template class ModelFactory<PHILIP_DIM, 5, RadFadType>;
template class ModelFactory<PHILIP_DIM, 8, RadFadType>;

} // Physics namespace
} // PHiLiP namespace

