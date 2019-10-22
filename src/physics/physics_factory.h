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
};


} // Physics namespace
} // PHiLiP namespace

#endif
