#include "parameters/all_parameters.h"
#include "parameters/parameters_artificial_dissipation.h"
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


template class ArtificialDissipationFactory<PHILIP_DIM, PHILIP_SPECIES,1>; 
template class ArtificialDissipationFactory<PHILIP_DIM, PHILIP_SPECIES,2>; 
template class ArtificialDissipationFactory<PHILIP_DIM, PHILIP_SPECIES,3>; 
template class ArtificialDissipationFactory<PHILIP_DIM, PHILIP_SPECIES,4>; 
template class ArtificialDissipationFactory<PHILIP_DIM, PHILIP_SPECIES,5>; 
template class ArtificialDissipationFactory<PHILIP_DIM, PHILIP_SPECIES,6>; 

} // namespace PHiLiP
