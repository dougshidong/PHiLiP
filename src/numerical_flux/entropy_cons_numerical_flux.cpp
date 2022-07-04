#include "entropy_cons_numerical_flux.hpp"
#include "ADTypes.hpp"

namespace PHiLiP {
namespace NumericalFlux {

template <int dim, int nstate, typename real>
std::array<real, nstate> EntropyConsNumFlux<dim,nstate,real>::evaluate_flux(
 const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_split_flux;

    conv_phys_split_flux = pde_physics->convective_numerical_split_flux (soln_int,soln_ext);

    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        real flux_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_dot_n += conv_phys_split_flux[s][d] * normal_int[d];
        }
        numerical_flux_dot_n[s] = flux_dot_n;
    }
    return numerical_flux_dot_n;
}
template class EntropyConsNumFlux<PHILIP_DIM, 1, double>;
template class EntropyConsNumFlux<PHILIP_DIM, 2, double>;
template class EntropyConsNumFlux<PHILIP_DIM, 3, double>;
template class EntropyConsNumFlux<PHILIP_DIM, 4, double>;
template class EntropyConsNumFlux<PHILIP_DIM, 5, double>;
template class EntropyConsNumFlux<PHILIP_DIM, 1, FadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 2, FadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 3, FadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 4, FadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 5, FadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 1, RadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 2, RadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 3, RadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 4, RadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 5, RadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 1, FadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 2, FadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 3, FadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 4, FadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 5, FadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 1, RadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 2, RadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 3, RadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 4, RadFadType >;
template class EntropyConsNumFlux<PHILIP_DIM, 5, RadFadType >;

}
}

