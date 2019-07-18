#include "split_form.h"
namespace PHiLiP
{

//SplitForm Factory
template <int dim, int nstate, typename real>
std::shared_ptr < SplitFormBase<dim, nstate, real> >
SplitFormFactory<dim,nstate,real>::create_SplitForm(Parameters::AllParameters::PartialDifferentialEquation pde_type)
{
	using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
	if (pde_type == PDE_enum::burgers_inviscid) {
		if constexpr (nstate == dim) return std::make_shared < SplitFormBurgers1D<dim,nstate,real> >(true,false); //what does true and false do?
	}
	std::cout << "Can't create SplitFormBase, invalid PDE type: " << pde_type << std::endl;
	assert(0==1 && "Can't create SplitFormBase, invalid PDE type");
	return nullptr;
}

} //PHiLiP namespace
