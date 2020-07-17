#ifndef __FIELD_H__
#define __FIELD_H__

#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

namespace PHiLiP {

namespace GridRefinement {

// object to store the anisotropic description of the field
template <int dim, typename real>
class Field 
{
public:
	// reinitialize the internal vector
	virtual void reinit(
		const unsigned int size) = 0;

	// reference for element size
	virtual real& scale(
		const unsigned int index) = 0;

	// setting the scale
	virtual void set_scale(
		const unsigned int index,
		const real         val) = 0;

	// getting the scale
	virtual real get_scale(
		const unsigned int index) = 0;

	// setting the anisotropic ratio
	virtual void set_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j,
		const real         ratio) = 0;

	// getting the anisotropic ratio
	virtual real get_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j) = 0;

	// setting the (unit) axis direction
	virtual void set_unit_axis(
		const unsigned int               index,
		const unsigned int               j,
		const dealii::Tensor<1,dim,real> unit_axis) = 0;

	// getting the (unit) axis direction
	virtual dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int index,
		const unsigned int j) = 0;

	// setting frame axis j (scaled) at index
	virtual void set_axis(
		const unsigned int               index,
		const unsigned int               j,
		const dealii::Tensor<1,dim,real> axis) = 0;

	// getting frame axis j (scaled) at index
	virtual dealii::Tensor<1,dim,real> get_axis(
		const unsigned int index,
		const unsigned int j) = 0;

	// get metric value at index
	virtual dealii::Tensor<2,dim,real> get_metric(
		const unsigned int index) = 0;
		
};

// isotropic element case (element scale only)
template <int dim, typename real>
class FieldIsotropic : public Field<dim,real>
{
public:
	// reinitialize the internal vector
	void reinit(
		const unsigned int size) override;

	// reference for element size
	real& scale(
		const unsigned int index) override;

	// setting the scale
	void set_scale(
		const unsigned int index,
		const real         val) override;

	// getting the scale
	real get_scale(
		const unsigned int index) override;

	// setting the anisotropic ratio
	void set_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j,
		const real         ratio) override;

	// getting the anisotropic ratio
	real get_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j) override;

	// setting the (unit) axis direction
	void set_unit_axis(
		const unsigned int               index,
		const unsigned int               j,
		const dealii::Tensor<1,dim,real> unit_axis) override;

	// getting the (unit) axis direction
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int index,
		const unsigned int j) override;

	// setting frame axis j (scaled) at index
	void set_axis(
		const unsigned int               index,
		const unsigned int               j,
		const dealii::Tensor<1,dim,real> axis) override;

	// getting frame axis j (scaled) at index
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int index,
		const unsigned int j) override;

	// get metric value at index
	dealii::Tensor<2,dim,real> get_metric(
		const unsigned int index) override;

private:
	// element data
	class ElementIsotropic
	{
	public:
		// element size
		real scale;
	};

	// vector of element data
	dealii::Vector<ElementIsotropic> field;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif //__FIELD_H__
