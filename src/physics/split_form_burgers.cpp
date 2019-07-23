#include "split_form.h"

namespace PHiLiP
{

namespace splitfunctions
{

namespace burgers1d
{

template <int dim, int nstate, typename real>
real F1<dim, nstate, real>::operator()(const std::array<real,nstate> &conservative_soln)
{
	return conservative_soln[0];
}

template <int dim, int nstate, typename real>
real F2<dim, nstate, real>::operator()(const std::array<real,nstate>)
{
	return 1.0;
}

template <int dim, int nstate, typename real>
real G1<dim, nstate, real>::operator()(const std::array<real,nstate> &conservative_soln)
{
	return conservative_soln[0];
}

template <int dim, int nstate, typename real>
real G2<dim, nstate, real>::operator()(const std::array<real,nstate> &conservative_soln)
{
	return conservative_soln[0] * conservative_soln[0] /2.;
}

} //burgers1d namespace

} //functions namespace

template <int dim, int nstate, typename real>
SplitFormBurgers1D<dim, nstate, real>::SplitFormBurgers1D()
{
	 assert (dim == 1);

	 SplitElement<dim,nstate,real> split_element;
	 std::vector<SplitElement<dim,nstate,real>> vector_of_split_elements;

	 split_element.alpha = 1./3.;
	 split_element.f = f1;
	 split_element.g = g1;

	 vector_of_split_elements.push_back(split_element);

	 split_element.alpha = 2./3.;
	 split_element.f = f2;
	 split_element.g = g2;

	 vector_of_split_elements.push_back(split_element);

	 SplitFormBase<dim, nstate,real>::split_convective_fluxes[0][0] = vector_of_split_elements;

	 vector_of_split_elements.clear();
}

template class SplitFormBurgers1D<PHILIP_DIM, 1, double>;
template class SplitFormBurgers1D<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class SplitFormBurgers1D<PHILIP_DIM, 2, double>;
template class SplitFormBurgers1D<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class SplitFormBurgers1D<PHILIP_DIM, 3, double>;
template class SplitFormBurgers1D<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class SplitFormBurgers1D<PHILIP_DIM, 4, double>;
template class SplitFormBurgers1D<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class SplitFormBurgers1D<PHILIP_DIM, 5, double>;
template class SplitFormBurgers1D<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

} //PHiLiP namespace


