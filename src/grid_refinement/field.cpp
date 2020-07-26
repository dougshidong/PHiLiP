#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>

#include "field.h"

namespace PHiLiP {

namespace GridRefinement {

/***** ***** ElementIsotropic ***** *****/

template <int dim, typename real>
real& ElementIsotropic<dim,real>::scale()
{
	return m_scale;
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::set_scale(
	const real val)
{
	m_scale = val;
}

template <int dim, typename real>
real ElementIsotropic<dim,real>::get_scale()
{
	return m_scale;
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::set_anisotropic_ratio(
	const std::array<real,dim>& /* ratio */)
{
	assert(0); // anisotropic ratio cannot be modified
}

template <int dim, typename real>
std::array<real,dim> ElementIsotropic<dim,real>::get_anisotropic_ratio()
{
	std::array<real,dim> anisotropic_ratio;

	for(unsigned int j = 0; j < dim; ++j)
		anisotropic_ratio[j] = get_anisotropic_ratio(j);

	return anisotropic_ratio;
}

template <int dim, typename real>
real ElementIsotropic<dim,real>::get_anisotropic_ratio(
	const unsigned int /* j */)
{
	return real(1.0);
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::set_unit_axis(
	const std::array<dealii::Tensor<1,dim,real>,dim>& /* unit_axis */)
{
	assert(0); // unit axis cannot be modified
}

template <int dim, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> ElementIsotropic<dim,real>::get_unit_axis()
{
	std::array<dealii::Tensor<1,dim,real>,dim> unit_axis;

	for(unsigned int j = 0; j < dim; ++j)
		unit_axis[j] = get_unit_axis(j);

	return unit_axis;
}

template <int dim, typename real>
dealii::Tensor<1,dim,real> ElementIsotropic<dim,real>::get_unit_axis(
	const unsigned int j)
{
	// getting unit vector in j^th coordinate
	dealii::Tensor<1,dim,real> u;
	u[j] = real(1.0);

	return u;
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::set_axis(
	const std::array<dealii::Tensor<1,dim,real>,dim>& /* axis */)
{
	assert(0); // axis cannot be modified
}

template <int dim, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> ElementIsotropic<dim,real>::get_axis()
{
	std::array<dealii::Tensor<1,dim,real>,dim> axis;

	for(unsigned int j = 0; j < dim; ++j)
		axis[j] = get_axis(j);

	return axis;
}

template <int dim, typename real>
dealii::Tensor<1,dim,real> ElementIsotropic<dim,real>::get_axis(
	const unsigned int j)
{
	return get_unit_axis(j) * get_scale();
}

template <int dim, typename real>
dealii::Tensor<2,dim,real> ElementIsotropic<dim,real>::get_metric()
{
	dealii::Tensor<2,dim,real> M;
	for(unsigned int i = 0; i < dim; ++i)
		M[i] = m_scale * get_unit_axis(i);

	return M;
}

/***** ***** ElementAnisotropic ***** *****/

template <int dim, typename real>
ElementAnisotropic<dim,real>::ElementAnisotropic()
{
	// sets values to default
	clear();
}

template <int dim, typename real>
real& ElementAnisotropic<dim,real>::scale()
{
	return m_scale;
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::set_scale(
	const real         val)
{
	m_scale = val;
}

template <int dim, typename real>
real ElementAnisotropic<dim,real>::get_scale()
{
	return m_scale;
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::set_anisotropic_ratio(
	const std::array<real,dim>& ratio)
{
	m_anisotropic_ratio = ratio;

	correct_unit_axis();
}

template <int dim, typename real>
real ElementAnisotropic<dim,real>::get_anisotropic_ratio(
	const unsigned int j)
{
	return m_anisotropic_ratio[j];
}

template <int dim, typename real>
std::array<real,dim> ElementAnisotropic<dim,real>::get_anisotropic_ratio()
{
	return m_anisotropic_ratio;
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::set_unit_axis(
	const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis)
{
	m_unit_axis = unit_axis;

	correct_element();
}

template <int dim, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> ElementAnisotropic<dim,real>::get_unit_axis()
{
	return m_unit_axis;
}

template <int dim, typename real>
dealii::Tensor<1,dim,real> ElementAnisotropic<dim,real>::get_unit_axis(
	unsigned int j)
{
	return m_unit_axis[j];
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::set_axis(
	const std::array<dealii::Tensor<1,dim,real>,dim>& axis)
{
	// reset length to 1 (unnormalized)
	m_scale = 1.0;

	// copy into unit_axis, reset the aspect ratio
	m_unit_axis = axis;
	for(unsigned int j = 0; j < dim; ++j)
		m_anisotropic_ratio[j] = 1.0;

	// correct and transfer values to other properties
	correct_element();
}

template <int dim, typename real>
dealii::Tensor<1,dim,real> ElementAnisotropic<dim,real>::get_axis(
	unsigned int j)
{
	return m_unit_axis[j] * m_anisotropic_ratio[j] * m_scale;
}

template <int dim, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> ElementAnisotropic<dim,real>::get_axis()
{
	std::array<dealii::Tensor<1,dim,real>,dim> axis;

	for(unsigned int j = 0; j < dim; ++j)
		axis[j] = get_axis(j);

	return axis;
}

template <int dim, typename real>
dealii::Tensor<2,dim,real> ElementAnisotropic<dim,real>::get_metric()
{
	dealii::Tensor<2,dim,real> M;

	for(unsigned int j = 0; j < dim; ++j)
		M[j] = m_scale * m_anisotropic_ratio[j] * m_unit_axis[j];

	return M;
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::clear()
{
	// define no element
	m_scale = 0.0;

	for(unsigned int j = 0; j < dim; ++j){
		// isotropic element
		m_anisotropic_ratio[j] = 1.0;

		// setting the unit axis to e_i
		m_unit_axis[j]    = dealii::Tensor<1,dim,real>();
		m_unit_axis[j][j] = 1.0;
	}

}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::correct_element()
{
	correct_unit_axis();
	correct_anisotropic_ratio();
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::correct_unit_axis()
{
	for(unsigned int j = 0; j < dim; ++j){
		// getting the axis length
		real length = m_unit_axis[j].norm();

		// applying change to anisotropic ratio and axis
		m_anisotropic_ratio[j] *= length;
		m_unit_axis[j]         /= length;
	}
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::correct_anisotropic_ratio()
{
	// getting the product of the ratios
	real product = 1.0;
	for(unsigned int j = 0; j < dim; ++j)
		product *= m_anisotropic_ratio[j];

	// determining the change that must be applied uniformly to each
	real alpha = pow(product, -1.0/dim);

	// applying this change to the ratios
	for(unsigned int j = 0; j < dim; ++j)
		m_anisotropic_ratio[j] *= alpha;

	// scaling the length
	m_scale *= alpha;
}

/***** ***** Field ***** *****/

template <int dim, typename real>
void Field<dim,real>::set_scale_vector_dealii(
	const dealii::Vector<real>& vec)
{
	for(unsigned int i = 0; i < this->size(); ++i)
		this->set_scale(i, vec[i]);
}

template <int dim, typename real>
void Field<dim,real>::set_scale_vector(
	const std::vector<real>& vec)
{
	for(unsigned int i = 0; i < this->size(); ++i)
		this->set_scale(i, vec[i]);
}

template <int dim, typename real>
dealii::Vector<real> Field<dim,real>::get_scale_vector_dealii()
{
	dealii::Vector<real> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_scale(i);

	return vec;
}

template <int dim, typename real>
std::vector<real> Field<dim,real>::get_scale_vector()
{
	std::vector<real> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_scale(i);

	return vec;
}

template <int dim, typename real>
void Field<dim,real>::set_axis_vector(
	const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& vec)
{
	for(unsigned int i = 0; i < this->size(); ++i)
		this->set_axis(i, vec[i]);
}

template <int dim, typename real>
std::vector<std::array<dealii::Tensor<1,dim,real>,dim>> Field<dim,real>::get_axis_vector()
{
	std::vector<std::array<dealii::Tensor<1,dim,real>,dim>> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_axis(i);

	return vec;
}

template <int dim, typename real>
std::vector<dealii::Tensor<1,dim,real>> Field<dim,real>::get_axis_vector(
	const unsigned int j)
{
	std::vector<dealii::Tensor<1,dim,real>> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_axis(i, j);

	return vec;
}

template <int dim, typename real>
std::vector<dealii::Tensor<2,dim,real>> Field<dim,real>::get_metric_vector()
{
	std::vector<dealii::Tensor<2,dim,real>> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_metric(i);

	return vec;
}

template <int dim, typename real>
dealii::SymmetricTensor<2,dim,real> Field<dim,real>::get_quadratic_metric(
	const unsigned int index)
{
	dealii::SymmetricTensor<2,dim,real> quadratic_metric;

	dealii::Tensor<2,dim,real> metric = this->get_metric(index);

	// looping over the upper triangular part
	for(unsigned int i = 0; i < dim; ++i){
		for(unsigned int j = i; j < dim; ++j){
			// assigning compoennts of A = M^T M from a_ij = v_i^T v_j where M = [v_1, ..., v_n]
			quadratic_metric[i][j] = scalar_product(metric[i], metric[j]);
		}
	}

	return quadratic_metric;
}

template <int dim, typename real>
std::vector<dealii::SymmetricTensor<2,dim,real>> Field<dim,real>::get_quadratic_metric_vector()
{
	std::vector<dealii::SymmetricTensor<2,dim,real>> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_quadratic_metric(i);

	return vec;
}

/***** ***** FieldInternal ***** *****/

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::reinit(
	const unsigned int size)
{
	field.resize(size);
}

template <int dim, typename real, typename ElementType>
unsigned int FieldInternal<dim,real,ElementType>::size()
{
	return field.size();
}

template <int dim, typename real, typename ElementType>
real& FieldInternal<dim,real,ElementType>::scale(
	const unsigned int index)
{
	return field[index].scale();
}

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::set_scale(
	const unsigned int index,
	const real         val)
{
	field[index].set_scale(val);
}

template <int dim, typename real, typename ElementType>
real FieldInternal<dim,real,ElementType>::get_scale(
	const unsigned int index)
{
	return field[index].get_scale();
}

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::set_anisotropic_ratio(
	const unsigned int index,
	const std::array<real,dim>& ratio)
{
	field[index].set_anisotropic_ratio(ratio);
}

template <int dim, typename real, typename ElementType>
std::array<real,dim> FieldInternal<dim,real,ElementType>::get_anisotropic_ratio(
	const unsigned int index)
{
	return field[index].get_anisotropic_ratio();
}

template <int dim, typename real, typename ElementType>
real FieldInternal<dim,real,ElementType>::get_anisotropic_ratio(
	const unsigned int index,
	const unsigned int j)
{
	return field[index].get_anisotropic_ratio(j);
}

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::set_unit_axis(
	const unsigned int                                index,
	const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis)
{
	field[index].set_unit_axis(unit_axis);
}

template <int dim, typename real, typename ElementType>
std::array<dealii::Tensor<1,dim,real>,dim> FieldInternal<dim,real,ElementType>::get_unit_axis(
	const unsigned int index)
{
	return field[index].get_unit_axis();
}


template <int dim, typename real, typename ElementType>
dealii::Tensor<1,dim,real> FieldInternal<dim,real,ElementType>::get_unit_axis(
	const unsigned int index,
	const unsigned int j)
{
	return field[index].get_unit_axis(j);
}

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::set_axis(
	const unsigned int                                index,
	const std::array<dealii::Tensor<1,dim,real>,dim>& axis)
{
	field[index].set_axis(axis);
}

template <int dim, typename real, typename ElementType>
std::array<dealii::Tensor<1,dim,real>,dim> FieldInternal<dim,real,ElementType>::get_axis(
	const unsigned int index)
{
	return field[index].get_axis();
}

template <int dim, typename real, typename ElementType>
dealii::Tensor<1,dim,real> FieldInternal<dim,real,ElementType>::get_axis(
	const unsigned int index,
	const unsigned int j)
{
	return field[index].get_axis(j);
}

template <int dim, typename real, typename ElementType>
dealii::Tensor<2,dim,real> FieldInternal<dim,real,ElementType>::get_metric(
	const unsigned int index)
{
	return field[index].get_metric();
}

template class Field <PHILIP_DIM, double>;

// FieldIsotropic
template class FieldInternal <PHILIP_DIM, double, ElementIsotropic<PHILIP_DIM, double>>;

// FieldAnisotropic
template class FieldInternal <PHILIP_DIM, double, ElementAnisotropic<PHILIP_DIM, double>>;

} // namespace GridRefinement

} // namespace PHiLiP
