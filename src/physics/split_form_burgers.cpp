#include "split_form.h"

namespace PHiLiP
{

template <int dim, int nstate, typename real>
double SplitFormBurgers1D<dim, nstate, real>::functions::F1::operator()(const std::array<double,nstate> &variables)
{
	return variables[0];
}

template <int dim, int nstate, typename real>
double SplitFormBurgers1D<dim, nstate, real>::functions::F2::operator()(const std::array<double,nstate> &variables)
{
	return 1.0;
}

template <int dim, int nstate, typename real>
double SplitFormBurgers1D<dim, nstate, real>::functions::G1::operator()(const std::array<double,nstate> &variables)
{
	return variables[0];
}

template <int dim, int nstate, typename real>
double SplitFormBurgers1D<dim, nstate, real>::functions::G2::operator()(const std::array<double,nstate> &variables)
{
	return variables[0] * variables[0] /2.;
}

template <int dim, int nstate, typename real>
SplitFormBurgers1D<dim, nstate, real>::SplitFormBurgers1D()
{
	 assert (dim == 1);

	 SplitElement<dim,nstate,real> split_element;
	 std::vector<SplitElement<dim,nstate,real>> vector_of_split_elements;

	 split_element.alpha = 1./3.;
	 split_element.f = functions::f1;
	 split_element.g = functions::g1;

	 vector_of_split_elements.push_back(split_element);

	 split_element.alpha = 2./3.;
	 split_element.f = functions::f2;
	 split_element.g = functions::g2;

	 vector_of_split_elements.push_back(split_element);

	 SplitFormBase<dim, nstate,real>::split_convective_fluxes[0] = vector_of_split_elements;

	 vector_of_split_elements.clear();
}

}
