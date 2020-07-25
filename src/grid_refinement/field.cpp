#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>

#include "field.h"

namespace PHiLiP {

namespace GridRefinement {

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
	const unsigned int                             j,
	const std::vector<dealii::Tensor<1,dim,real>>& vec)
{
	for(unsigned int i = 0; i < this->size(); ++i)
		this->set_axis(i, j, vec[i]);
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
	const unsigned int                /* index */,
	const unsigned int                /* j */,
	const dealii::Tensor<1,dim,real>& /* unit_axis */)
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
	const unsigned int                /* index */,
	const unsigned int                /* j */,
	const dealii::Tensor<1,dim,real>& /* axis */)
{
	assert(0); // axis cannot be modified
}


template <int dim, typename real>
dealii::Tensor<1,dim,real> FieldIsotropic<dim,real>::get_axis(
	const unsigned int               index,
	const unsigned int               j)
{
	return get_unit_axis(index, j) * get_scale(index);
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
