#ifndef __PHYSICS_FACTORY__
#define __PHYSICS_FACTORY__

#include "parameters/all_parameters.h"
#include "physics.h"

namespace PHiLiP {
namespace Physics {
/// Create specified physics as PhysicsBase object 
/** Factory design pattern whose job is to create the correct physics
 */
template <int dim, int nstate, typename real>
class PhysicsFactory
{
public:
    /// Factory to return the correct physics given input file.
    static std::shared_ptr< PhysicsBase<dim,nstate,real> >
        create_Physics(const Parameters::AllParameters *const parameters_input);

    /// Factory to return the correct physics given input file and a specified PDE type
    static std::shared_ptr< PhysicsBase<dim,nstate,real> >
        create_Physics(
            const Parameters::AllParameters *const parameters_input,
            const Parameters::AllParameters::PartialDifferentialEquation pde_type);

private:
    /// Factory to return the correct physics model, i.e. when PDE_type==physics_model, given input file
    static std::shared_ptr< PhysicsBase<dim,nstate,real> >
        create_Physics_Model(
            const Parameters::AllParameters                           *const parameters_input,
            const dealii::Tensor<2,3,double>                          diffusion_tensor,
            std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);
};


} // Physics namespace
} // PHiLiP namespace

#endif
