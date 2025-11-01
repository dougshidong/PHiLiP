#include "parameters/all_parameters.h"
#include "parameters/parameters_artificial_dissipation.h"
#include <boost/preprocessor/seq/for_each.hpp>
#include <deal.II/base/tensor.h>
#include "artificial_dissipation_factory.h"
#include "artificial_dissipation.h"

namespace PHiLiP {

template <int dim, int nspecies, int nstate>
std::shared_ptr < ArtificialDissipationBase<dim,nspecies,nstate> >
ArtificialDissipationFactory<dim,nspecies,nstate> ::create_artificial_dissipation(const Parameters::AllParameters *const parameters_input)
{
    using artificial_dissipation_enum = Parameters::ArtificialDissipationParam::ArtificialDissipationType;
    artificial_dissipation_enum arti_dissipation_type = parameters_input->artificial_dissipation_param.artificial_dissipation_type;

    switch (arti_dissipation_type)
    {
        case artificial_dissipation_enum::laplacian:
        {
            return std::make_shared<LaplacianArtificialDissipation<dim,nspecies,nstate>>(parameters_input);
            break;
        }

        case artificial_dissipation_enum::physical:
        {
            if constexpr(dim+2==nstate)
            {
                std::cout<<"Physical Artifical Dissipation pointer created"<<std::endl;
                return std::make_shared<PhysicalArtificialDissipation<dim,nspecies,nstate>>(parameters_input);
            }
            break;
        }

        case artificial_dissipation_enum::enthalpy_conserving_laplacian:
        {
            if constexpr(dim+2==nstate)
            {
                std::cout<<"Enthalpy Conserving Laplacian Artifical Dissipation pointer created"<<std::endl;
                return std::make_shared<EnthalpyConservingArtificialDissipation<dim,nspecies,nstate>>(parameters_input);
            }
            break;
        }

    }

    assert(0==1 && "Cannot create artificial dissipation due to an invalid artificial dissipation type specified for the problem"); 
    return nullptr;
}

#if PHILIP_SPECIES==1
    // Define a sequence of indices representing the range [1, 6]
    #define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)

    // Define a macro to instantiate MyTemplate for a specific index
    #define INSTANTIATE_ADFactory(r, data, index) \
    template class ArtificialDissipationFactory <PHILIP_DIM, PHILIP_SPECIES, index>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_ADFactory, _, POSSIBLE_NSTATE)
#endif
} // namespace PHiLiP
