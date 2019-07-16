#include <iostream>
#include <functional>
#include <vector>
#include "physics.h"
#include "parameters/all_parameters.h"
namespace  PHiLiP
{

namespace splitform
{

template <int dim, int nstate, typename real>
class SplitFormBase
{
public:
	SplitFormBase() = delete;
	//now we define an array (of size nstate) of a vector of split form functions.
	//This is because each component has an unknown number of split functions (hence the use of std::vector), and each equation has nstate components (hence the use of std::array).
	typedef std::array < std::vector< std::function< std::array<double,nstate>(std::array<double,nstate>) > >,nstate > split_list;
	split_list split_form_functions;
};

template <int dim, int nstate, typename real>
class SplitFormBurgers : public SplitFormBase<dim, nstate, real>
{
public:
	SplitFormBurgers() = delete;
	SplitFormBurgers(Physics::PhysicsBase<dim, nstate, real> *physics);
};

template <int dim, int nstate, typename real>
class SplitFormFactory
{
public:
	static std::shared_ptr< SplitFormBase<dim, nstate, real> >
		create_SplitForm(Parameters::AllParameters::PartialDifferentialEquation pde_type);
};

} //splitform namespace

} //PHiLiP namespace
