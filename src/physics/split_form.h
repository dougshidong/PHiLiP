#include <iostream>
#include <functional>
#include <vector>
#include <assert.h>
#include "physics.h"
#include "parameters/all_parameters.h"

#define USING_SPLIT_FORM

#ifndef __SPLIT_FORM_H__
#define __SPLIT_FORM_H__

namespace  PHiLiP
{



namespace splitfunctions // can't put namespaces inside classes, but nested classes are possible
{

namespace burgers1d
{
	template <int dim, int nstate, typename real>
	class F1 //the reasons we implement functions as functors is so we can pass them as objects as opposed to functions (which are more messy to deal with)
	{
	public:
		F1(){};
		real operator()(const std::array<real,nstate> &conservative_soln);
	};

	template <int dim, int nstate, typename real>
	class F2
	{
	public:
		F2(){};
		real operator()(const std::array<real,nstate> /*&conservative_soln*/);
	};

	template <int dim, int nstate, typename real>
	class G1
	{
	public:
		G1(){};
		real operator()(const std::array<real,nstate> &conservative_soln);
	};

	template <int dim, int nstate, typename real>
	class G2
	{
	public:
		G2(){};
		real operator()(const std::array<real,nstate> &conservative_soln);
};
}


}


template<int dim, int nstate, typename real>
class SplitElement
{
public:
	SplitElement() {};
	real alpha;
	std::function<real(std::array<real,nstate>)> f;
	std::function<real(std::array<real,nstate>)> g;
};

template <int dim, int nstate, typename real>
class SplitFormBase
{
public:
	SplitFormBase() {};
	//now we define an array (of size nstate) of a vector of split form functions.
	//This is because each component has an unknown number of split functions (hence the use of std::vector), and each equation has nstate components (hence the use of std::array).
	typedef std::array< std::array<std::vector<SplitElement<dim,nstate,real>>,nstate>, dim > split_list;
	split_list split_convective_fluxes;
};


template <int dim, int nstate, typename real>
class SplitFormBurgers1D : public SplitFormBase<dim, nstate, real> //technically this isn't templated on nstate and dim because we need to know them in order to implement our split functions
{
public:
	SplitFormBurgers1D() = delete;
	SplitFormBurgers1D(Parameters::AllParameters::PartialDifferentialEquation /*pde_type*/);
	splitfunctions::burgers1d::F1<dim, nstate, real> f1;
	splitfunctions::burgers1d::F2<dim, nstate, real> f2;
	splitfunctions::burgers1d::G1<dim, nstate, real> g1;
	splitfunctions::burgers1d::G2<dim, nstate, real> g2;
};

template <int dim, int nstate, typename real>
class SplitFormFactory
{
public:
	static std::shared_ptr< SplitFormBase<dim, nstate, real> >
		create_SplitForm(Parameters::AllParameters::PartialDifferentialEquation pde_type);
};



} //PHiLiP namespace

#endif
