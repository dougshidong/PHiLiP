#include <iostream>
#include <functional>
#include <vector>
#include <assert.h>
#include "physics.h"
#include "parameters/all_parameters.h"
namespace  PHiLiP
{

template<int dim, int nstate, typename real>
class SplitElement
{
public:
	SplitElement() {};
	//SplitElement(double alph); //NEED TO DO PROPER TEMPLATE INSTANTIATION
	double alpha;
	std::function<double(std::array<double,nstate>)> f;
	std::function<double(std::array<double,nstate>)> g;
};

template <int dim, int nstate, typename real>
class SplitFormBase
{
public:
	SplitFormBase() {};
	//now we define an array (of size nstate) of a vector of split form functions.
	//This is because each component has an unknown number of split functions (hence the use of std::vector), and each equation has nstate components (hence the use of std::array).
	typedef std::array<std::vector<SplitElement<dim,nstate,real>>,nstate> split_list;
	split_list split_convective_fluxes;
};


template <int dim, int nstate, typename real>
class SplitFormBurgers1D : public SplitFormBase<dim, nstate, real> //technically this isn't templated on nstate and dim because we need to know them in order to implement our split functions
{
public:
	SplitFormBurgers1D();

	class functions // can't put namespaces inside classes, but nested classes are possible
	{
	public:
		functions() {};
		class F1 //the reasons we implement functions as functors is so we can pass them as objects as opposed to functions (which are more messy to deal with)
		{
		public:
			F1(){};
			double operator()(const std::array<double,nstate> &variables);
		};

		class F2
		{
		public:
			F2(){};
			double operator()(const std::array<double,nstate> &variables);
		};

		class G1
		{
		public:
			G1(){};
			double operator()(const std::array<double,nstate> &variables);
		};

		class G2
		{
		public:
			G2(){};
			double operator()(const std::array<double,nstate> &variables);
		};

		F1 f1;
		F2 f2;
		G1 g1;
		G2 g2;
	};
};



template <int dim, int nstate, typename real>
class SplitFormFactory
{
public:
	static std::shared_ptr< SplitFormBase<dim, nstate, real> >
		create_SplitForm(Parameters::AllParameters::PartialDifferentialEquation pde_type);
};



} //PHiLiP namespace
