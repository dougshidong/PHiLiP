#include <ostream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/geometry_info.h>

#include "field.h"

namespace PHiLiP {

namespace GridRefinement {

/***** ***** Element ***** *****/
template <int dim, typename real>
typename Element<dim,real>::ChordList Element<dim,real>::get_chord_list(
	const typename Element<dim,real>::VertexList& vertices)
{

	// example notation from 2D (for nodes and chords) 
    /*      ^ chord_list[1]
            |
        2---+---3
        |   |   |
        +---o---+---> chord_list[0]
        |   |   |
        0---+---1
    */

	// construcing the list of chords
	Element<dim,real>::ChordList chord_list;

	// to determine distance from face center to face center
	// can use the difference of average node positions on each face
	// taking average (divide by number of face nodes) after summation

	// looping over the nodes and adding either their +/- weighting to the corresponding chords
	for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){

		// each vertex should have an impact on the chord of each axis
		for(unsigned int i = 0; i < dim; ++i){

			// adding either positive or negative weighting
			// can be determined by sign of i^th bit in vertex id binary
			if(vertex>>i % 2 == 0){

				// lies on +ve direction of axis
				chord_list[i] += vertices[vertex];

			}else{

				// lies on -ve face of axis
				chord_list[i] -= vertices[vertex];

			}

		}

	}

	// applying averaging for each node position (based on number on each face)
	for(unsigned int i = 0; i < dim; ++i)
		chord_list[i] /= dealii::GeometryInfo<dim>::vertices_per_face;

	// chord vectors should be fully computed
	return chord_list;

}

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
real ElementIsotropic<dim,real>::get_scale() const 
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

template <int dim, typename real>
dealii::Tensor<2,dim,real> ElementIsotropic<dim,real>::get_inverse_metric()
{
	// building the rotation matrix
	dealii::Tensor<2,dim,real> R;
	for(unsigned int i = 0; i < dim; ++i)
		R[i] = get_unit_axis(i);

	// assuming R^-1 = R^T (orthogonal)
	dealii::Tensor<2,dim,real> RT = transpose(R);

	// assembling the inverse metric based on
	// M = 1/h diag(1/r_i,...) R(-\theta)
	dealii::Tensor<2,dim,real> M;
	for(unsigned int i = 0; i < dim; ++i)
		M[i] = (1.0/m_scale) * RT[i];

	return M;
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::set_cell_internal(
	const typename Element<dim,real>::VertexList& vertices)
{
	// to be consistent with anisotropic case using the average from chord lengths
	typename Element<dim,real>::ChordList chord_list = Element<dim,real>::get_chord_list(vertices);
	
	// based on \Prod{L_i} = \Prod{h r_i} = h^dim \Prod{r_i}
	// where the product of axes anisotropy should be 1.
	real product = 1.0;
	for(unsigned int i = 0; i < dim; ++i)
		product *= chord_list[i].norm();

	// setting the scale from h (ignoring axes alignment)
	m_scale = pow(product, -1.0/dim);
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::set_anisotropy(
	const std::array<real,dim>&                       /* derivative_value */,
	const std::array<dealii::Tensor<1,dim,real>,dim>& /* derivative_direction */,
	const unsigned int                                /* order */)
{
	// invalid, cannot set anisotropy on isotropic cell
	assert(0);
}

template <int dim, typename real>
void ElementIsotropic<dim,real>::apply_anisotropic_limit(
	const real /* anisotropic_ratio_min */,
	const real /* anisotropic_ratio_max */)
{
	// invalid, cannot set anisotropy on isotropic cell
	assert(0);
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
real ElementAnisotropic<dim,real>::get_scale() const
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
dealii::Tensor<2,dim,real> ElementAnisotropic<dim,real>::get_inverse_metric()
{
	// building the rotation matrix
	dealii::Tensor<2,dim,real> R;
	for(unsigned int i = 0; i < dim; ++i)
		R[i] = get_unit_axis(i);

	// assuming R^-1 = R^T (orthogonal)
	dealii::Tensor<2,dim,real> RT = transpose(R);

	// assembling the inverse metric based on
	// M = 1/h diag(1/r_i,...) R(-\theta)
	dealii::Tensor<2,dim,real> M;
	for(unsigned int i = 0; i < dim; ++i)
		M[i] = (1.0)/(m_anisotropic_ratio[i]*m_scale) * RT[i];

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

	// perform sorting on the anisotropic ratios
	sort_anisotropic_ratio();
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::sort_anisotropic_ratio()
{
	
	// checking the sorting of the ansitropic ratio
	using anisopair = std::pair< real, dealii::Tensor<1,dim,real> >;
	std::array<anisopair, dim > sort_array;

	// storing the anisotropic ratio and their unit vectors
	for(unsigned int j = 0; j < dim; ++j)
		sort_array[j] = std::make_pair(
			m_anisotropic_ratio[j],
			m_unit_axis[j]);

	// performing the sort
	std::sort(sort_array.begin(), sort_array.end(), [](
		anisopair left,
		anisopair right)
	{
		return left.first > right.first;
	});

	// copying back into the corresponding array 
	for(unsigned int j = 0; j < dim; ++j){
		m_anisotropic_ratio[j] = sort_array[j].first;
		m_unit_axis[j]         = sort_array[j].second;
	}

}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::set_cell_internal(
	const typename Element<dim,real>::VertexList& vertices)
{
	// to be consistent with anisotropic case using the average from chord lengths
	typename Element<dim,real>::ChordList chord_list = Element<dim,real>::get_chord_list(vertices);
	
	// setting the axes to the chord values and correcting
	clear();

	// overall scale is 1.0 with unit axes as the unscaled chords
	m_scale = 1.0;
	m_unit_axis = chord_list;

	// correcting to fix scaling issues
	correct_element();
}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::set_anisotropy(
	const std::array<real,dim>&                       derivative_value,
	const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
	const unsigned int                                order)
{
	// currently only implemented for dim == 2
	assert(dim == 2);

	// std::cout << "d1 = " << derivative_value[0] << ", v1 = [" << derivative_direction[0] << "]" << std::endl;
	// std::cout << "d2 = " << derivative_value[1] << ", v2 = [" << derivative_direction[1] << "]" << std::endl;
	// std::cout << std::endl;

	// derivative value and direction should be an ordered set
	
	// computing the product of all
	real product = 1.0;
	for(unsigned int i = 0; i < dim; ++i)
		product *= derivative_value[i];

	real denominator = pow(product, 1.0/dim);

	// computing anisotropy A_i/(\Prod A)^{1/dim}
	std::array<real,dim> rho;
	for(unsigned int i = 0; i < dim; ++i)
		rho[i] = derivative_value[i]/denominator;

	// std::cout << "rho1 = " << rho[0] << std::endl;
	// std::cout << "rho2 = " << rho[1] << std::endl;
	// std::cout << std::endl;

	// anisotropy in each axis becomes \rho^{-1/(p+1)}
	// keeping direction values as is 
	// (anisotropic ratios should be swapped, doing this by removing - sign)
	std::array<real,dim> anisotropic_ratio;
	for(unsigned int i = 0; i < dim; ++i)
		anisotropic_ratio[i] = pow(rho[i], -1.0/order);

	// std::cout << "aniso1 = " << anisotropic_ratio[0] << std::endl;
	// std::cout << "aniso2 = " << anisotropic_ratio[1] << std::endl;
	// std::cout << std::endl;

	
	// setting values for the cell
	m_anisotropic_ratio = anisotropic_ratio;
	m_unit_axis         = derivative_direction;

	// correcting
	correct_element();

}

template <int dim, typename real>
void ElementAnisotropic<dim,real>::apply_anisotropic_limit(
	const real anisotropic_ratio_min,
	const real anisotropic_ratio_max)
{
	// boolean to keep track of if any terms have been updated
	bool changed = false;

	// loops assume that refinements can chain together
	// this requires that the anisotropic ratios are ordered in descending axis length

	// looping through in descending order to limit upper axes
	for(unsigned int index = 0; index < dim; ++index){
		// using the loop index
		int j = index;

		if(m_anisotropic_ratio[j] > anisotropic_ratio_max){
			// check, this should never occur on the last axis
			assert(index != dim-1);

			// capping the value to this axis and redistributing to all subsequent axes
			real factor = pow(m_anisotropic_ratio[j]/anisotropic_ratio_max, 1.0/(dim-1.0-index));
			changed = true;

			// setting to the new value (the upper limit)
			m_anisotropic_ratio[j] = anisotropic_ratio_max;

			// looping through for redistribution
			for(unsigned int index_internal = index+1; index_internal < dim; ++index_internal){

				// getting the corresponding index (same)
				int j_internal = index_internal;

				// applying the factor change
				m_anisotropic_ratio[j_internal] *= factor;

			}

		}
	}

	// looping through in ascending order to limit the lower axes
	for(unsigned int index = 0; index < dim; ++index){

		// reversing the index
		int j = dim-1-index;

		if(m_anisotropic_ratio[j] < anisotropic_ratio_min){
			// check, this should never occur on the last axis
			assert(index != dim-1);

			// capping the value to this axis and redistributing to all subsequent axes
			real factor = pow(m_anisotropic_ratio[j]/anisotropic_ratio_min, 1.0/(dim-1.0-index));
			changed = true;

			// setting to the new value (the upper limit)
			m_anisotropic_ratio[j] = anisotropic_ratio_min;

			// looping through for redistribution
			for(unsigned int index_internal = index+1; index_internal < dim; ++index_internal){

				// getting the corresponding reversed index
				int j_internal = dim-1-index_internal;

				// applying the factor change
				m_anisotropic_ratio[j_internal] *= factor;

			}

		}
	}

	// if values have changed, correcting the anisotropic ratios just in case of rounding errors
	if(changed)
		correct_anisotropic_ratio();

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
dealii::Vector<real> Field<dim,real>::get_scale_vector_dealii() const
{
	dealii::Vector<real> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_scale(i);

	return vec;
}

template <int dim, typename real>
std::vector<real> Field<dim,real>::get_scale_vector() const
{
	std::vector<real> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_scale(i);

	return vec;
}

template <int dim, typename real>
dealii::Vector<real> Field<dim,real>::get_max_anisotropic_ratio_vector_dealii()
{
	dealii::Vector<real> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i){
		// getting the cell anisotropic ratios
		std::array<real,dim> anisotropic_ratio = get_anisotropic_ratio(i);

		// finding the maximum value from iterators for the cell
		vec[i] = *std::max_element(anisotropic_ratio.begin(), anisotropic_ratio.end());
	}

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

	for(unsigned int i = 0; i < this->size(); ++i){
		// // temp
		// std::cout << "metric[" << i << "] = "
		// 	<< ", alpha = " << this->get_scale(i)
		// 	<< ", r1 = " << this->get_anisotropic_ratio(i, 0)
		// 	<< ", v1 = {" << this->get_unit_axis(i, 0) 
		// 	<< "}, r2 = " << this->get_anisotropic_ratio(i, 1)
		// 	<< ", v2 = {" << this->get_unit_axis(i, 1) << "}" << std::endl;
		
		vec[i] = this->get_metric(i);
	}

	return vec;
}

template <int dim, typename real>
std::vector<dealii::Tensor<2,dim,real>> Field<dim,real>::get_inverse_metric_vector()
{
	std::vector<dealii::Tensor<2,dim,real>> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i){
		// // temp
		// std::cout << "metric[" << i << "] = "
		// 	<< ", 1/alpha = " << (1.0/this->get_scale(i))
		// 	<< ", 1/r1 = " << (1.0/this->get_anisotropic_ratio(i, 0))
		// 	<< ", v1 = {" << this->get_unit_axis(i, 0) 
		// 	<< "}, 1/r2 = " << (1.0/this->get_anisotropic_ratio(i, 1))
		// 	<< ", v1 = {" << this->get_unit_axis(i, 1) << "}" << std::endl;
		
		vec[i] = this->get_inverse_metric(i);
	}

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

template <int dim, typename real>
dealii::SymmetricTensor<2,dim,real> Field<dim,real>::get_inverse_quadratic_metric(
	const unsigned int index)
{
	dealii::SymmetricTensor<2,dim,real> inverse_quadratic_metric;
	dealii::Tensor<2,dim,real> inverse_metric = this->get_inverse_metric(index);

	// // temp
	// std::cout << "metric[" << index << "] = "
	// 	<< ", 1/alpha = " << (1.0/this->get_scale(index))
	// 	<< ", 1/r1 = " << (1.0/this->get_anisotropic_ratio(index, 0))
	// 	<< ", v1 = {" << this->get_unit_axis(index, 0) 
	// 	<< "}, 1/r2 = " << (1.0/this->get_anisotropic_ratio(index, 1))
	// 	<< ", v1 = {" << this->get_unit_axis(index, 1) << "}" << std::endl;

	// std::cout << "Vinv = " << inverse_metric << std::endl;
	// std::cout << "Vinv[0] = " << inverse_metric[0] << std::endl;
	// std::cout << "Vinv[1] = " << inverse_metric[1] << std::endl;

	// needed for proper component eval, to get the columns with inverse_metric[i] instead of the rows
	inverse_metric = transpose(inverse_metric);

	// looping over the upper triangular part
	for(unsigned int i = 0; i < dim; ++i){
		for(unsigned int j = i; j < dim; ++j){
			// assigning compoennts of A = M^T M from a_ij = v_i^T v_j where M = [v_1, ..., v_n]
			inverse_quadratic_metric[i][j] = scalar_product(inverse_metric[i], inverse_metric[j]);
		}
	}

	// std::cout << "Vinv^T Vinv = " << inverse_quadratic_metric << std::endl << std::endl;

	return inverse_quadratic_metric;
}

template <int dim, typename real>
std::vector<dealii::SymmetricTensor<2,dim,real>> Field<dim,real>::get_inverse_quadratic_metric_vector()
{
	std::vector<dealii::SymmetricTensor<2,dim,real>> vec(this->size());

	for(unsigned int i = 0; i < this->size(); ++i)
		vec[i] = this->get_inverse_quadratic_metric(i);

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
unsigned int FieldInternal<dim,real,ElementType>::size() const
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
	const unsigned int index) const 
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

template <int dim, typename real, typename ElementType>
dealii::Tensor<2,dim,real> FieldInternal<dim,real,ElementType>::get_inverse_metric(
	const unsigned int index)
{
	return field[index].get_inverse_metric();
}

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::set_anisotropy(
	const dealii::DoFHandler<dim>&                                 dof_handler,
	const std::vector<std::array<real,dim>>&                       derivative_value,
	const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& derivative_direction,
	const int                                                      relative_order)
{
	// sizes must match
	assert(size() == dof_handler.get_triangulation().n_active_cells());
	assert(size() == derivative_value.size());
	assert(size() == derivative_direction.size());

	// looping through the cells
	for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
		if(!cell->is_locally_owned()) continue;

		// getting the index
		const unsigned int index = cell->active_cell_index();

		// getting the order
		const unsigned int order = cell->active_fe_index() + relative_order;

		// performing the call to update the anisotropy
		field[index].set_anisotropy(
			derivative_value[index],
			derivative_direction[index],
			order);
	}

}

template <int dim, typename real, typename ElementType>
void FieldInternal<dim,real,ElementType>::apply_anisotropic_limit(
	const real anisotropic_ratio_min,
	const real anisotropic_ratio_max)
{

	for(unsigned int index = 0; index < this->size(); ++index)
		field[index].apply_anisotropic_limit(
			anisotropic_ratio_min,
			anisotropic_ratio_max);

}

template <int dim, typename real, typename ElementType>
std::ostream& FieldInternal<dim,real,ElementType>::serialize(
	std::ostream& os) const 
{
	for(unsigned int index = 0; index < this->size(); ++index){
		// std::cout << "writing element index = " << index << std::endl;
		// std::cout << "with size = " << field[index].get_scale() << std::endl;
		os << field[index] << std::endl;
	}

	return os;
}

template class Field <PHILIP_DIM, double>;

// FieldIsotropic
template class FieldInternal <PHILIP_DIM, double, ElementIsotropic<PHILIP_DIM, double>>;

// FieldAnisotropic
template class FieldInternal <PHILIP_DIM, double, ElementAnisotropic<PHILIP_DIM, double>>;

} // namespace GridRefinement

} // namespace PHiLiP
