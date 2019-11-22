#include "parameters/all_parameters.h"

#include "physics_factory.h"
#include "manufactured_solution.h"
#include "physics.h"
#include "convection_diffusion.h"
#include "burgers.h"
#include "euler.h"
#include "mhd.h"

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
        = ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(parameters_input);

    if (pde_type == PDE_enum::advection || pde_type == PDE_enum::advection_vector) {
        if constexpr (nstate<=2) return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(true,false,manufactured_solution_function);
    } else if (pde_type == PDE_enum::diffusion) {
        if constexpr (nstate==1) return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(false,true,manufactured_solution_function);
    } else if (pde_type == PDE_enum::convection_diffusion) {
        if constexpr (nstate==1) return std::make_shared < ConvectionDiffusion<dim,nstate,real> >(true,true,manufactured_solution_function);
    } else if (pde_type == PDE_enum::burgers_inviscid) {
        if constexpr (nstate==dim) return std::make_shared < Burgers<dim,nstate,real> >(true,false,manufactured_solution_function);
    } else if (pde_type == PDE_enum::euler) {
        if constexpr (nstate==dim+2) {
            return std::make_shared < Euler<dim,nstate,real> > (parameters_input->euler_param.ref_length 
                                                               ,parameters_input->euler_param.gamma_gas
                                                               ,parameters_input->euler_param.mach_inf
                                                               ,parameters_input->euler_param.angle_of_attack
                                                               ,parameters_input->euler_param.side_slip_angle
                                                               ,manufactured_solution_function);
        }

    } else if (pde_type == PDE_enum::mhd) {
        if constexpr (nstate == 8) return std::make_shared < MHD<dim,nstate,real> > (parameters_input->euler_param.gamma_gas,manufactured_solution_function);
    }
    std::cout << "Can't create PhysicsBase, invalid PDE type: " << pde_type << std::endl;
    assert(0==1 && "Can't create PhysicsBase, invalid PDE type");
    return nullptr;
}

template class PhysicsFactory<PHILIP_DIM, 1, double>;
template class PhysicsFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class PhysicsFactory<PHILIP_DIM, 2, double>;
template class PhysicsFactory<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class PhysicsFactory<PHILIP_DIM, 3, double>;
template class PhysicsFactory<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class PhysicsFactory<PHILIP_DIM, 4, double>;
template class PhysicsFactory<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class PhysicsFactory<PHILIP_DIM, 5, double>;
template class PhysicsFactory<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;
template class PhysicsFactory<PHILIP_DIM, 8, double>;
template class PhysicsFactory<PHILIP_DIM, 8, Sacado::Fad::DFad<double> >;


} // Physics namespace
} // PHiLiP namespace

