#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

#include "field.h"

namespace PHiLiP {

namespace GridRefinement {

/***** ***** Field ***** *****/

template <int dim, typename real>
dealii::Vector<real> Field<dim,real>::get_scale_vector()
{
	dealii::Vector<real> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_scale(i);

	return vec;
}

template <int dim, typename real>
void Field<dim,real>::set_scale_vector(
	dealii::Vector<real>& vec)
{
	for(unsigned int i = 0; i < this->size(); ++i)
		this->set_scale(i, vec[i]);
}

/***** ***** FieldIsotropic ***** *****/

template <int dim, typename real>
void FieldIsotropic<dim,real>::reinit(
	const unsigned int size)
{
	field.resize(size);
}

template <int dim, typename real>
unsigned int FieldIsotropic<dim,real>::size()
{
	return field.size();
}

template <int dim, typename real>
real& FieldIsotropic<dim,real>::scale(
	const unsigned int index)
{
	return field[index].scale;
}

template <int dim, typename real>
void FieldIsotropic<dim,real>::set_scale(
	const unsigned int index,
	const real         val)
{
	field[index].scale = val;
}

template <int dim, typename real>
real FieldIsotropic<dim,real>::get_scale(
	const unsigned int index)
{
	return field[index].scale;
}

template <int dim, typename real>
void FieldIsotropic<dim,real>::set_anisotropic_ratio(
	const unsigned int /* index */,
	const unsigned int /* j */,
	const real         /* ratio */)
{
	assert(0); // anisotropic ratio cannot be modified
}

template <int dim, typename real>
real FieldIsotropic<dim,real>::get_anisotropic_ratio(
	const unsigned int /* index */,
	const unsigned int /* j */)
{
	return real(1.0);
}

template <int dim, typename real>
void FieldIsotropic<dim,real>::set_unit_axis(
	const unsigned int               /* index */,
	const unsigned int               /* j */,
	const dealii::Tensor<1,dim,real> /* unit_axis */)
{
	assert(0); // unit axis cannot be modified
}

template <int dim, typename real>
dealii::Tensor<1,dim,real> FieldIsotropic<dim,real>::get_unit_axis(
	const unsigned int /* index */,
	const unsigned int j)
{
	// getting unit vector in j^th coordinate
	dealii::Tensor<1,dim,real> u;
	u[j] = real(1.0);

	return u;
}

template <int dim, typename real>
void FieldIsotropic<dim,real>::set_axis(
	const unsigned int               /* index */,
	const unsigned int               /* j */,
	const dealii::Tensor<1,dim,real> /* axis */)
{
	assert(0); // axis cannot be modified
}


template <int dim, typename real>
dealii::Tensor<1,dim,real> FieldIsotropic<dim,real>::get_axis(
	const unsigned int               /* index */,
	const unsigned int               /* j */)
{
	assert(0); // unit axis cannot be modified
}

template <int dim, typename real>
dealii::Tensor<2,dim,real> FieldIsotropic<dim,real>::get_metric(
	const unsigned int index)
{
	dealii::Tensor<2,dim,real> M;
	for(unsigned int i = 0; i < dim; ++i)
		M[i][i] = field[index].scale;

	return M;
}

/***** ***** FieldAnisotropic ***** *****/



template class Field <PHILIP_DIM, double>;
template class FieldIsotropic <PHILIP_DIM, double>;

} // namespace GridRefinement

} // namespace PHiLiP
