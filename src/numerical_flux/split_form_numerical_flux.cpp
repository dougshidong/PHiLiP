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
		//std::cout << "evaluating the split form flux" <<std::endl;
		using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
#if 0
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
	        numerical_flux_dot_n[s] = conv_phys_split_flux[s]*normal_int - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
	    }
	   // std::cout << "about to return split num flux" <<std::endl;
#endif

//#if 0
    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = pde_physics->convective_flux (soln_int);
    conv_phys_flux_ext = pde_physics->convective_flux (soln_ext);
    RealArrayVector flux_avg;
    for (int s=0; s<nstate; s++) {
        flux_avg[s] = 0.5*(conv_phys_flux_int[s] + conv_phys_flux_ext[s]);
    }
   // RealArrayVector conv_phys_split_flux;

   // conv_phys_split_flux = pde_physics->convective_numerical_split_flux (soln_int,soln_ext);
  //  RealArrayVector flux_avg = array_average<nstate, dealii::Tensor<1,dim,real>> (conv_phys_flux_int, conv_phys_flux_ext);
   // const real conv_max_eig = 1.0/12.0 * (soln_ext[0] - soln_int[0]);
    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = flux_avg[s]*normal_int - 1.0/12.0 * pow((soln_ext[s]-soln_int[s]),2.0);
       // numerical_flux_dot_n[s] = conv_phys_split_flux[s]*normal_int - conv_max_eig * (soln_ext[s]-soln_int[s]);
    }
//#endif

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
