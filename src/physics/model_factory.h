#ifndef __MODEL_FACTORY__
#define __MODEL_FACTORY__

#include "parameters/all_parameters.h"
#include "model.h"

namespace PHiLiP {
namespace Physics {
/// Create specified model as ModelBase object 
/** Factory design pattern whose job is to create the correct model
 */
template <int dim, int nstate, typename real>
class ModelFactory
{
public:
    /// Factory to return the correct physics given input file and grid spacing
    static std::shared_ptr< ModelBase<dim,nstate,real> >
        create_Model(const Parameters::AllParameters  *const parameters_input,
                     const double                     grid_spacing);
};


} // Physics namespace
} // PHiLiP namespace

#endif
