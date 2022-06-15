#include "split_form_numerical_flux.hpp"
#include "ADTypes.hpp"
//#include <Sacado.hpp>
//#include <deal.II/differentiation/ad/sacado_product_types.h>

namespace PHiLiP {
namespace NumericalFlux {

template <int dim, int nstate, typename real>
std::array<real, nstate> SplitFormNumFlux<dim,nstate,real>::evaluate_flux(
 const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
 {
  //std::cout << "evaluating the split form flux" <<std::endl;
  using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
     RealArrayVector conv_phys_split_flux;

     conv_phys_split_flux = pde_physics->convective_numerical_split_flux (soln_int,soln_ext);
     //std::cout << "done evaluating the conv num split flux" <<std::endl;

     const real conv_max_eig_int = pde_physics->max_convective_eigenvalue(soln_int);
    // std::cout << "1st eig" << std::endl;
     const real conv_max_eig_ext = pde_physics->max_convective_eigenvalue(soln_ext);
     //std::cout << "2nd eig" << std::endl;
     const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);

    // std::cout << "obtained the max eig" <<std::endl;
     // Scalar dissipation
     std::array<real, nstate> numerical_flux_dot_n;
     for (int s=0; s<nstate; s++) {
         real flux_dot_n = 0.0;
         for (int d=0; d<dim; ++d) {
             flux_dot_n += conv_phys_split_flux[s][d] * normal_int[d];
         }
         numerical_flux_dot_n[s] = flux_dot_n - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
     }
    // std::cout << "about to return split num flux" <<std::endl;
     return numerical_flux_dot_n;
 }

template class SplitFormNumFlux<PHILIP_DIM, 1, double>;
template class SplitFormNumFlux<PHILIP_DIM, 2, double>;
template class SplitFormNumFlux<PHILIP_DIM, 3, double>;
template class SplitFormNumFlux<PHILIP_DIM, 4, double>;
template class SplitFormNumFlux<PHILIP_DIM, 5, double>;
template class SplitFormNumFlux<PHILIP_DIM, 1, FadType >;
template class SplitFormNumFlux<PHILIP_DIM, 2, FadType >;
template class SplitFormNumFlux<PHILIP_DIM, 3, FadType >;
template class SplitFormNumFlux<PHILIP_DIM, 4, FadType >;
template class SplitFormNumFlux<PHILIP_DIM, 5, FadType >;
template class SplitFormNumFlux<PHILIP_DIM, 1, RadType >;
template class SplitFormNumFlux<PHILIP_DIM, 2, RadType >;
template class SplitFormNumFlux<PHILIP_DIM, 3, RadType >;
template class SplitFormNumFlux<PHILIP_DIM, 4, RadType >;
template class SplitFormNumFlux<PHILIP_DIM, 5, RadType >;
template class SplitFormNumFlux<PHILIP_DIM, 1, FadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 2, FadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 3, FadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 4, FadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 5, FadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 1, RadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 2, RadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 3, RadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 4, RadFadType >;
template class SplitFormNumFlux<PHILIP_DIM, 5, RadFadType >;

}
}
