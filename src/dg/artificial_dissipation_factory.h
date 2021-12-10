#ifndef __ARTIFICIAL_DISSIPATION_FACTORY__
#define __ARTIFICIAL_DISSIPATION_FACTORY__

#include "parameters/all_parameters.h"
#include "artificial_dissipation.h"


namespace PHiLiP
{
/// Creates artificial dissipation pointer
template<int dim, int nstate>
class ArtificialDissipationFactory
{
    public:
    /// Creates artificial dissipation type depending on input parameters.
    static std::shared_ptr<ArtificialDissipationBase<dim,nstate>> create_artificial_dissipation(const Parameters::AllParameters *const parameters_input);
};

} // PHiLiP namespace

#endif


