#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

#include <deal.II/base/tensor.h>

#include "ADTypes.hpp"

#include "physics_factory.h"
#include "manufactured_solution.h"
#include "physics.h"
#include "convection_diffusion.h"
#include "burgers.h"
#include "burgers_rewienski.h"
#include "euler.h"
#include "mhd.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::shared_ptr < PhysicsBase<dim,nstate,real> >
PhysicsFactory<dim,nstate,real>
::create_Physics(const Parameters::AllParameters *const parameters_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    PDE_enum pde_type = parameters_input->pde_type;

    // generating the manufactured solution from the manufactured solution factory
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> >  manufactured_solution_function 
        = ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(parameters_input, nstate);

    // setting the diffusion tensor and advection vectors from parameters (if needed)
    const dealii::Tensor<2,3,double> diffusion_tensor      = parameters_input->manufactured_convergence_study_param.manufactured_solution_param.diffusion_tensor;
    const dealii::Tensor<1,3,double> advection_vector      = parameters_input->manufactured_convergence_study_param.manufactured_solution_param.advection_vector;
    const double                     diffusion_coefficient = parameters_input->manufactured_convergence_study_param.manufactured_solution_param.diffusion_coefficient;

    if (pde_type == PDE_enum::advection || pde_type == PDE_enum::advection_vector) {
        if constexpr (nstate<=2) 
            return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(
                true, false,
                diffusion_tensor, advection_vector, diffusion_coefficient,
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::diffusion) {
        if constexpr (nstate==1) 
            return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(
                false, true,
                diffusion_tensor, advection_vector, diffusion_coefficient,
                manufactured_solution_function,
                parameters_input->test_type);
    } else if (pde_type == PDE_enum::convection_diffusion) {
        if constexpr (nstate==1) 
            return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(
                true, true,
                diffusion_tensor, advection_vector, diffusion_coefficient,
                manufactured_solution_function,
                parameters_input->test_type);
    } else if (pde_type == PDE_enum::burgers_inviscid) {
        if constexpr (nstate==dim) 
            return std::make_shared < Burgers<dim,nstate,real> >(
                true, false,
                diffusion_tensor, 
                manufactured_solution_function,
                parameters_input->test_type);
    } else if (pde_type == PDE_enum::burgers_rewienski) {
        if constexpr (nstate==dim)
            return std::make_shared < BurgersRewienski<dim,nstate,real> >(
                    parameters_input->reduced_order_param.rewienski_a,
                    parameters_input->reduced_order_param.rewienski_b,
                    parameters_input->reduced_order_param.rewienski_manufactured_solution,
                    true,
                    false,
                    diffusion_tensor,
                    manufactured_solution_function);
    } else if (pde_type == PDE_enum::euler) {
        if constexpr (nstate==dim+2) {
            return std::make_shared < Euler<dim,nstate,real> > (
                parameters_input->euler_param.ref_length,
                parameters_input->euler_param.gamma_gas,
                parameters_input->euler_param.mach_inf,
                parameters_input->euler_param.angle_of_attack,
                parameters_input->euler_param.side_slip_angle,
                diffusion_tensor, 
                manufactured_solution_function);
        }
    } else if (pde_type == PDE_enum::mhd) {
        if constexpr (nstate == 8) 
            return std::make_shared < MHD<dim,nstate,real> > (
                parameters_input->euler_param.gamma_gas,
                diffusion_tensor, 
                manufactured_solution_function);
    } else if (pde_type == PDE_enum::navier_stokes) {
        if constexpr (nstate==dim+2) {
            return std::make_shared < NavierStokes<dim,nstate,real> > (
                parameters_input->euler_param.ref_length,
                parameters_input->euler_param.gamma_gas,
                parameters_input->euler_param.mach_inf,
                parameters_input->euler_param.angle_of_attack,
                parameters_input->euler_param.side_slip_angle,
                parameters_input->navier_stokes_param.prandtl_number,
                parameters_input->navier_stokes_param.reynolds_number_inf,
                diffusion_tensor, 
                manufactured_solution_function);
        }
    } else {
        // prevent warnings for dim=3,nstate=4, etc.
        (void) diffusion_tensor;
        (void) advection_vector;
        (void) diffusion_coefficient;
    }
    std::cout << "Can't create PhysicsBase, invalid PDE type: " << pde_type << std::endl;
    assert(0==1 && "Can't create PhysicsBase, invalid PDE type");
    return nullptr;
}

template class PhysicsFactory<PHILIP_DIM, 1, double>;
template class PhysicsFactory<PHILIP_DIM, 2, double>;
template class PhysicsFactory<PHILIP_DIM, 3, double>;
template class PhysicsFactory<PHILIP_DIM, 4, double>;
template class PhysicsFactory<PHILIP_DIM, 5, double>;
template class PhysicsFactory<PHILIP_DIM, 8, double>;

template class PhysicsFactory<PHILIP_DIM, 1, FadType >;
template class PhysicsFactory<PHILIP_DIM, 2, FadType >;
template class PhysicsFactory<PHILIP_DIM, 3, FadType >;
template class PhysicsFactory<PHILIP_DIM, 4, FadType >;
template class PhysicsFactory<PHILIP_DIM, 5, FadType >;
template class PhysicsFactory<PHILIP_DIM, 8, FadType >;

template class PhysicsFactory<PHILIP_DIM, 1, RadType >;
template class PhysicsFactory<PHILIP_DIM, 2, RadType >;
template class PhysicsFactory<PHILIP_DIM, 3, RadType >;
template class PhysicsFactory<PHILIP_DIM, 4, RadType >;
template class PhysicsFactory<PHILIP_DIM, 5, RadType >;
template class PhysicsFactory<PHILIP_DIM, 8, RadType >;

template class PhysicsFactory<PHILIP_DIM, 1, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 2, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 3, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 4, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 5, FadFadType >;
template class PhysicsFactory<PHILIP_DIM, 8, FadFadType >;

template class PhysicsFactory<PHILIP_DIM, 1, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 2, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 3, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 4, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 5, RadFadType >;
template class PhysicsFactory<PHILIP_DIM, 8, RadFadType >;



} // Physics namespace
} // PHiLiP namespace
