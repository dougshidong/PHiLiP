#include "split_form_numerical_flux.h"
#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>

namespace PHiLiP {
namespace NumericalFlux {

template <int dim, int nstate, typename real>
std::array<real, nstate> SplitFormNumFlux<dim,nstate,real>::evaluate_flux(
	const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
	{
		using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
	    RealArrayVector conv_phys_split_flux;

	    conv_phys_split_flux = pde_physics->convective_numerical_split_flux (soln_int,soln_ext);


	    const real conv_max_eig_int = pde_physics->max_convective_eigenvalue(soln_int);
	    const real conv_max_eig_ext = pde_physics->max_convective_eigenvalue(soln_ext);
	    const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
	    // Scalar dissipation
	    std::array<real, nstate> numerical_flux_dot_n;
	    for (int s=0; s<nstate; s++) {
	        numerical_flux_dot_n[s] = conv_phys_split_flux[s]*normal_int - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
	    }

	    return numerical_flux_dot_n;
	}

template class SplitFormNumFlux<PHILIP_DIM, 1, double>;
template class SplitFormNumFlux<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class SplitFormNumFlux<PHILIP_DIM, 2, double>;
template class SplitFormNumFlux<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class SplitFormNumFlux<PHILIP_DIM, 3, double>;
template class SplitFormNumFlux<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class SplitFormNumFlux<PHILIP_DIM, 4, double>;
template class SplitFormNumFlux<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class SplitFormNumFlux<PHILIP_DIM, 5, double>;
template class SplitFormNumFlux<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

}
}
