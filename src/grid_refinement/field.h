#ifndef __FIELD_H__
#define __FIELD_H__

#include <ostream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/hp/dof_handler.h>

namespace PHiLiP {

namespace GridRefinement {

// element base class that accesses field information at a point (corresponding element)
template <int dim, typename real>
class Element
{
public:
	// reference for element size
	virtual real& scale() = 0;

	// setting the scale
	virtual void set_scale(
		const real val) = 0;

	// getting the scale
	virtual real get_scale() = 0;

	// setting the anisotropic ratio
	virtual void set_anisotropic_ratio(
		const std::array<real,dim>& ratio) = 0;

	// getting the anisotropic ratio (array)
	virtual std::array<real,dim> get_anisotropic_ratio() = 0;

	// getting the anisotropic ratio
	virtual real get_anisotropic_ratio(
		const unsigned int j) = 0;

	// setting the (unit) axis direction
	virtual void set_unit_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) = 0;

	// getting the (unit) axis direction (array)
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis() = 0;

	// getting the (unit) axis direction
	virtual dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int j) = 0;

	// setting frame axis j (scaled) at index
	virtual void set_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) = 0;

	// getting frame axis j (scaled) at index (array)
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_axis() = 0;

	// getting frame axis j (scaled) at index
	virtual dealii::Tensor<1,dim,real> get_axis(
		const unsigned int j) = 0;

	// get metric value at index
	virtual dealii::Tensor<2,dim,real> get_metric() = 0;

	// defining the type of the vertices and chord lists for easier definition
	using VertexList = std::array<dealii::Tensor<1,dim,real>, dealii::GeometryInfo<dim>::vertices_per_cell>;
	using ChordList  = std::array<dealii::Tensor<1,dim,real>, dim>;

protected:
	// gets the chord list from an input set of vertices
	ChordList get_chord_list(
		const VertexList& vertices);

	// sets the element based on a list of vertices
	// internal function for handling set_cell below
	virtual void set_cell_internal(
		const VertexList& vertices) = 0;

public:

	// sets the Element based on the input cell (from current mesh)
	template <typename DoFCellAccessorType>
	void set_cell(
		const DoFCellAccessorType& cell)
	{
		VertexList vertices;

		for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
			vertices[vertex] = cell->vertex(vertex);

		set_cell_internal(vertices);
	}

	// Dolejsi's anisotropy from reconstructed directional derivatives (reconstruct_poly)
	virtual void set_anisotropy(
		const std::array<real,dim>&                       derivative_value,
		const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
		const unsigned int                                order) = 0;

	// limits the anisotropic ratios to a given bandwidth
	virtual void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max) = 0;

};

// isotropic element case (element scale only)
template <int dim, typename real>
class ElementIsotropic : public Element<dim,real>
{
public:
	// reference for element size
	real& scale() override;

	// setting the scale
	void set_scale(
		const real val) override;

	// getting the scale
	real get_scale() override;

	// setting the anisotropic ratio
	void set_anisotropic_ratio(
		const std::array<real,dim>& ratio) override;

	// getting the anisotropic ratio (array)
	std::array<real,dim> get_anisotropic_ratio() override;

	// getting the anisotropic ratio
	real get_anisotropic_ratio(
		const unsigned int j) override;

	// setting the (unit) axis direction
	void set_unit_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) override;

	// getting the (unit) axis direction (array)
	std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis() override;

	// getting the (unit) axis direction
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int j) override;

	// setting frame axis j (scaled) at index
	void set_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) override;

	// getting frame axis j (scaled) at index (array)
	std::array<dealii::Tensor<1,dim,real>,dim> get_axis() override;

	// getting frame axis j (scaled) at index
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int j) override;

	// get metric value at index
	dealii::Tensor<2,dim,real> get_metric() override;

protected:
	// sets the element based on a list of vertices
	// internal function for handling set_cell below
	void set_cell_internal(
		const typename Element<dim,real>::VertexList& vertices) override;

public:
	// Dolejsi's anisotropy from reconstructed directional derivatives (reconstruct_poly)
	void set_anisotropy(
		const std::array<real,dim>&                       derivative_value,
		const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
		const unsigned int                                order) override;

	// limits the anisotropic ratios to a given bandwidth
	void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max);

	// output to ostream
	friend std::ostream& operator<<(
		std::ostream&                     os,
		const ElementIsotropic<dim,real>& element) 
	{
		os << "Isotropic element with properties:" << std::endl
		   << '\t' << "m_scale = " << element.m_scale << std::endl;

		return os;
	}

private:
	// element size
	real m_scale;
};

// anisotropic element case (stores frame axes)
template <int dim, typename real>
class ElementAnisotropic : public Element<dim,real>
{
public:
	// constructor
	ElementAnisotropic();

	// reference for element size
	real& scale() override;

	// setting the scale
	void set_scale(
		const real val) override;

	// getting the scale
	real get_scale() override;

	// setting the anisotropic ratio
	void set_anisotropic_ratio(
		const std::array<real,dim>& ratio) override;

	// getting the anisotropic ratio (array)
	std::array<real,dim> get_anisotropic_ratio() override;

	// getting the anisotropic ratio
	real get_anisotropic_ratio(
		const unsigned int j) override;

	// setting the (unit) axis direction
	void set_unit_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) override;

	// getting the (unit) axis direction (array)
	std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis() override;

	// getting the (unit) axis direction
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int j) override;

	// setting frame axis j (scaled) at index
	void set_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) override;

	// getting frame axis j (scaled) at index (array)
	std::array<dealii::Tensor<1,dim,real>,dim> get_axis() override;

	// getting frame axis j (scaled) at index
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int j) override;

	// get metric value at index
	dealii::Tensor<2,dim,real> get_metric() override;

protected:
	// sets the element based on a list of vertices
	// internal function for handling set_cell below
	void set_cell_internal(
		const typename Element<dim,real>::VertexList& vertices) override;

public:
	// Dolejsi's anisotropy from reconstructed directional derivatives (reconstruct_poly)
	void set_anisotropy(
		const std::array<real,dim>&                       derivative_value,
		const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
		const unsigned int                                order) override;

	// limits the anisotropic ratios to a given bandwidth
	void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max);

	// output to ostream
	friend std::ostream& operator<<(
		std::ostream&                       os,
		const ElementAnisotropic<dim,real>& element)
	{
		os << "Anisotropic element with properties:" << std::endl
		   << '\t' << "m_scale = " << element.m_scale << std::endl
		   << '\t' << "m_anisotropic_ratio = {" << element.m_anisotropic_ratio[0];
		
		for(unsigned int i = 1; i < dim; ++i)
			os << ", " << element.m_anisotropic_ratio[i];

		os << "}" << std::endl
		   << '\t' << "m_unit_axis = {" << element.m_unit_axis[0];

		for(unsigned int i = 1; i < dim; ++i)
			os << ", " << element.m_unit_axis[i];

		os << "}" << std::endl;

		return os;
	}

private:

	// resets the element
	void clear();

	// corrects the values stored internally
	void correct_element();

	// renormalize the unit_axis by factoring any scaling into anisotropic ratio
	// correct_anisotropic_ratio() should be called immediately afterward
	void correct_unit_axis();

	// adjust the ratio values s.t. product of anisotropic ratios equals 1 
	void correct_anisotropic_ratio();

	// sorts the anisotropic ratio values (and corresponding unit axis)
	void sort_anisotropic_ratio();

	// element size
	real m_scale;

	// axes ratios
	std::array<real, dim> m_anisotropic_ratio;

	// axes directions
	std::array<dealii::Tensor<1,dim,real>, dim> m_unit_axis;
};

// object to store the anisotropic description of the field
template <int dim, typename real>
class Field 
{
public:
	// reinitialize the internal vector
	virtual void reinit(
		const unsigned int size) = 0;

	// returns the internal vector size
	virtual unsigned int size() const = 0;

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

	// setting the scale vector (dealii::Vector)
	void set_scale_vector_dealii(
		const dealii::Vector<real>& vec);
	
	// setting the scale vector (std::vector)
	void set_scale_vector(
		const std::vector<real>& vec);

	// getting the scale vector (dealii::Vector)
	dealii::Vector<real> get_scale_vector_dealii();

	// getting the scale vector (std::vector)
	std::vector<real> get_scale_vector();

	// setting the anisotropic ratio
	virtual void set_anisotropic_ratio(
		const unsigned int          index,
		const std::array<real,dim>& ratio) = 0;

	// getting the anisotropic ratio
	virtual std::array<real,dim> get_anisotropic_ratio(
		const unsigned int index) = 0;

	// getting the anisotropic ratio
	virtual real get_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j) = 0;

	// gets the dealii vector of max anisotropic ratio for each cell
	dealii::Vector<real> get_max_anisotropic_ratio_vector_dealii();

	// setting the (unit) axis direction
	virtual void set_unit_axis(
		const unsigned int                                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) = 0;

	// getting the (unit) axis direction
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis(
		const unsigned int index) = 0;

	// getting the (unit) axis direction
	virtual dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int index,
		const unsigned int j) = 0;

	// setting frame axis j (scaled) at index
	virtual void set_axis(
		const unsigned int                                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) = 0;

	// getting frame axis j (scaled) at index
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_axis(
		const unsigned int index) = 0;

	// getting frame axis j (scaled) at index
	virtual dealii::Tensor<1,dim,real> get_axis(
		const unsigned int index,
		const unsigned int j) = 0;

	// setting frame axis j (scaled) vector (std::vector)
	void set_axis_vector(
		const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& vec);

	// getting frame axis j (scaled) vector (std::vector)
	std::vector<std::array<dealii::Tensor<1,dim,real>,dim>> get_axis_vector();

	// getting frame axis j (scaled) vector (std::vector)
	std::vector<dealii::Tensor<1,dim,real>> get_axis_vector(
		const unsigned int j);

	// get metric value at index
	virtual dealii::Tensor<2,dim,real> get_metric(
		const unsigned int index) = 0;

	// getting the metric vector (std::vector)
	std::vector<dealii::Tensor<2,dim,real>> get_metric_vector();

	// get riemanian quadratic metric \mathcal{M} = M^T M
	dealii::SymmetricTensor<2,dim,real> get_quadratic_metric(
		const unsigned int index);

	// gets the riemanian quadratic metric \mathcal{M} = M^T M in vector format
	std::vector<dealii::SymmetricTensor<2,dim,real>> get_quadratic_metric_vector();

	// defining the associated DofHandler type
	using DoFHandlerType = dealii::hp::DoFHandler<dim>;

	// asigns the field based on an input DoFHandlerType
	virtual void set_cell(
		const DoFHandlerType& dof_handler) = 0;

	// Dolejsi's anisotropy from reconstructed directional derivatives (reconstruct_poly)
	virtual void set_anisotropy(
		const dealii::hp::DoFHandler<dim>&                             dof_handler,
		const std::vector<std::array<real,dim>>&                       derivative_value,
		const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& derivative_direction,
		const int                                                      relative_order) = 0;
	
	// limits the anisotropic ratios to a given bandwidth
	virtual void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max) = 0;

	// performs the internal call to writing to an ostream from the field
	virtual std::ostream& serialize(
		std::ostream& os) const = 0;

	// outputs to ostream
	friend std::ostream& operator<<(
		std::ostream&          os,
		const Field<dim,real>& field)
	{
		return field.serialize(os);
	}

};

// wrapper to hide element type
template <int dim, typename real, typename ElementType>
class FieldInternal : public Field<dim,real>
{
public:
	// reinitialize the internal vector
	void reinit(
		const unsigned int size);

	// returns the internal vector size
	unsigned int size() const;

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
		const unsigned int          index,
		const std::array<real,dim>& ratio) override;

	// getting the anisotropic ratio
	std::array<real,dim> get_anisotropic_ratio(
		const unsigned int index) override;

	// getting the anisotropic ratio
	real get_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j) override;

	// setting the (unit) axis direction
	void set_unit_axis(
		const unsigned int                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) override;

	// getting the (unit) axis direction
	std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis(
		const unsigned int index) override;

	// getting the (unit) axis direction
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int index,
		const unsigned int j) override;

	// setting frame axis j (scaled) at index
	void set_axis(
		const unsigned int                                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) override;

	// getting frame axis j (scaled) at index
	std::array<dealii::Tensor<1,dim,real>,dim> get_axis(
		const unsigned int index) override;

	// getting frame axis j (scaled) at index
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int index,
		const unsigned int j) override;

	// get metric value at index
	dealii::Tensor<2,dim,real> get_metric(
		const unsigned int index) override;

	// asigns the field based on an input DoFHandlerType
	void set_cell(
		const typename Field<dim,real>::DoFHandlerType& dof_handler) override
	{
		reinit(dof_handler.get_triangulation().n_active_cells());

		for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
			if(cell->is_locally_owned())
				field[cell->active_cell_index()].set_cell(cell);
	}

	// Dolejsi's anisotropy from reconstructed directional derivatives (reconstruct_poly)
	void set_anisotropy(
		const dealii::hp::DoFHandler<dim>&                             dof_handler,
		const std::vector<std::array<real,dim>>&                       derivative_value,
		const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& derivative_direction,
		const int                                                      relative_order) override;

	// limits the anisotropic ratios to a given bandwidth
	void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max) override;

	// performs the internal call to writing to an ostream from the field
	std::ostream& serialize(
		std::ostream& os) const override;

private:
	// vector of element data
	std::vector<ElementType> field;
};

// isotropic element case (element scale only)
template <int dim, typename real>
using FieldIsotropic = FieldInternal<dim,real,ElementIsotropic<dim,real>>;

// anisotropic element case (stores frame axes)
template <int dim, typename real>
using FieldAnisotropic = FieldInternal<dim,real,ElementAnisotropic<dim,real>>;

} // namespace GridRefinement

} // namespace PHiLiP

#endif //__FIELD_H__
