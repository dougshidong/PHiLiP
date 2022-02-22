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

/// Element class
/** This provides a base class for incorporating both isotropic (size) and anisotropic (size, anisotropy orientation)
  * in controlling unstructured mesh adaptation methods. Provides functions for setting and accesing the local values 
  * for a given cell. A collection of elements is contained in the corresponding Field class (with respective extensions
  * for anisotropy). Note: setting anisotropic properties in the isotropic case will assert(0) and make no changes.
  */
template <int dim, typename real>
class Element
{
public:
	/// Reference for element size
	/** Allows direct read/write of scale of mean element axis.
	  * Measure of element (length, area, volume) will be \f$scale^{dim}\f$
	  */
	virtual real& scale() = 0;

	/// Set the scale for the element
	virtual void set_scale(
		const real val) = 0;

	/// Get the scale for the element
	virtual real get_scale() const = 0;

	/// Set the anisotropic ratio for each reference axis
	/** Requires array of order matching axis definition. 
	  * Each reference axis will have length \f$l = \alpha * scale\f$.
	  * Note: does nothing in the isotropic case. 
	  */
	virtual void set_anisotropic_ratio(
		const std::array<real,dim>& ratio) = 0;

	/// Get the anisotropic ratio of each reference axis as an array
	/** Note: equals 1 for each axis in isotropic case.
	  */ 
	virtual std::array<real,dim> get_anisotropic_ratio() = 0;

	/// Get the anisotropic ratio corresponding to the \f$j^{th}\f$ reference axis
	/** Note: equals 1 for each axis in isotropic case.
	  */ 
	virtual real get_anisotropic_ratio(
		const unsigned int j) = 0;

	/// Set the unit axes of the element
	/** Requires an ordered array of dealii::Tensor with vector axes.
	  * If none unit values are provided, will be rescaled and factored in to anisotropic ratios.
	  * Note: does nothing in the isotropic case.
	  */
	virtual void set_unit_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) = 0;

	/// Get unit axis directions as an array
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis() = 0;

	/// Get the \f$j^{th}\f$ reference axis
	virtual dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int j) = 0;

	/// Set the scaled local frame axes based on vector set (length and direction)
	/** Describe the axes of reference parrelogram or parrelopiped at point.
	  * Note: does nothing in the isotropic case.
	  */
	virtual void set_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) = 0;

	/// Get the array of axes of the local frame field \f$(v_1,v_2,v_3)\f$
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_axis() = 0;

	/// Get the vector corresponding to the \f$j^{th}\f$ frame axis \f$v_j\f$
	virtual dealii::Tensor<1,dim,real> get_axis(
		const unsigned int j) = 0;

	/// Get metric matrix at point describing mapping from reference element
	/** In 2D orthorgonal case, \f$V = [v,w] = h * R(\theta) * \mathrm{diag}{\rho,1/\rho}\f$.
	  * Under transformation, order of axes is maintained with \f$(i,j,k)\f$ vectors mapping to \f$(v_1,v_2,v_3)\f$
	  */ 
	virtual dealii::Tensor<2,dim,real> get_metric() = 0;

	///. Get inverse metric matrix for the reference element
	/** In 2D orthogonal case, \f$V = 1/h * \mathrm{diag}{\rho,1/\rho} * R(-\theta)\f$.
	  */ 
	virtual dealii::Tensor<2,dim,real> get_inverse_metric() = 0;

	/// Type alias for array of vertex coordinates for element
	using VertexList = std::array<dealii::Tensor<1,dim,real>, dealii::GeometryInfo<dim>::vertices_per_cell>;
	/// Type alias for array of chord veectors (face center to face center) in Deal.II ordering
	using ChordList  = std::array<dealii::Tensor<1,dim,real>, dim>;

protected:
	/// Get the chord list from an input set of vertices
	ChordList get_chord_list(
		const VertexList& vertices);

	/// Set element to match geometry of input vertex set
	/** Vertices describe the tensor product element in Deal.II ordering.
	  * Internal function used in handling of set_cell().
	  */
	virtual void set_cell_internal(
		const VertexList& vertices) = 0;

public:

	/// Set the Element based on the input cell (from current mesh)
	/** Templated on the mesh/dof type of the input cell from DoFHandler.
	  */
	template <typename DoFCellAccessorType>
	void set_cell(
		const DoFCellAccessorType& cell)
	{
		VertexList vertices;

		for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
			vertices[vertex] = cell->vertex(vertex);

		set_cell_internal(vertices);
	}

	/// Set anisotropy from reconstructed directional derivatives
	/** Based on Dolejsi's method for simplices, uses values obtained from
	  * reconstructed polynomial on neighbouring cell patch.
	  */
	virtual void set_anisotropy(
		const std::array<real,dim>&                       derivative_value,
		const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
		const unsigned int                                order) = 0;

	/// Limits the anisotropic ratios to a given bandwidth
	/** First finds ratio above max, redistributes length change to maintain constant volume and scale.
	  * Then the process is repeated with a lower bound.
	  * Note: does nothing in the isotropic case.
	  */ 
	virtual void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max) = 0;

};

/// Isotropic element class
/** Specialization of the element type for the isotropic remeshing case.
  * Contains only the scale of local element (size field) .
  * A collection of elements is contained in the corresponding Field class.
  * Note: virtual functions controlling axes or anisotropic ratio do nothing, assert(0).
  */
template <int dim, typename real>
class ElementIsotropic : public Element<dim,real>
{
public:
	/// Reference for element size
	/** Allows direct read/write of scale of mean element axis.
	  * Measure of element (length, area, volume) will be \f$scale^{dim}\f$
	  */
	real& scale() override;

	/// Set the scale for the element
	void set_scale(
		const real val) override;

	/// Get the scale for the element
	real get_scale() const override;

	/// Set the anisotropic ratio for each reference axis
	/** Requires array of order matching axis definition. 
	  * Each reference axis will have length \f$l = \alpha * scale\f$.
	  * Note: does nothing in the isotropic case. 
	  */
	void set_anisotropic_ratio(
		const std::array<real,dim>& ratio) override;

	/// Get the anisotropic ratio of each reference axis as an array
	/** Note: equals 1 for each axis in isotropic case.
	  */ 
	std::array<real,dim> get_anisotropic_ratio() override;

	/// Get the anisotropic ratio corresponding to the \f$j^{th}\f$ reference axis
	/** Note: equals 1 for each axis in isotropic case.
	  */ 
	real get_anisotropic_ratio(
		const unsigned int j) override;

	/// Set the unit axes of the element
	/** Requires an ordered array of dealii::Tensor with vector axes.
	  * If none unit values are provided, will be rescaled and factored in to anisotropic ratios.
	  * Note: does nothing in the isotropic case.
	  */
	void set_unit_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) override;

	/// Get unit axis directions as an array
	std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis() override;

	/// Get the \f$j^{th}\f$ reference axis
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int j) override;

	/// Set the scaled local frame axes based on vector set (length and direction)
	/** Describe the axes of reference parrelogram or parrelopiped at point.
	  * Note: does nothing in the isotropic case.
	  */
	void set_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) override;

	/// Get the array of axes of the local frame field \f$(v_1,v_2,v_3)\f$
	std::array<dealii::Tensor<1,dim,real>,dim> get_axis() override;

	/// Get the vector corresponding to the \f$j^{th}\f$ frame axis \f$v_j\f$
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int j) override;

	/// Get metric matrix at point describing mapping from reference element
	/** In 2D orthorgonal case, \f$V = [v,w] = h * R(\theta) * \mathrm{diag}{\rho,1/\rho}\f$.
	  * Under transformation, order of axes is maintained with \f$(i,j,k)\f$ vectors mapping to \f$(v_1,v_2,v_3)\f$
	  */
	dealii::Tensor<2,dim,real> get_metric() override;

	/// Get inverse metric matrix for the reference element
	/** In 2D orthogonal case, \f$V = 1/h * \mathrm{diag}{\rho,1/\rho} * R(-\theta)\f$.
	  */ 
	dealii::Tensor<2,dim,real> get_inverse_metric() override;

protected:
	/// Set element to match geometry of input vertex set
	/** Vertices describe the tensor product element in Deal.II ordering.
	  * Internal function used in handling of set_cell().
	  */
	void set_cell_internal(
		const typename Element<dim,real>::VertexList& vertices) override;

public:
	/// Set anisotropy from reconstructed directional derivatives
	/** Based on Dolejsi's method for simplices, uses values obtained from
	  * reconstructed polynomial on neighbouring cell patch.
	  */
	void set_anisotropy(
		const std::array<real,dim>&                       derivative_value,
		const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
		const unsigned int                                order) override;

	/// Limits the anisotropic ratios to a given bandwidth
	/** First finds ratio above max, redistributes length change to maintain constant volume and scale.
	  * Then the process is repeated with a lower bound.
	  * Note: does nothing in the isotropic case.
	  */ 
	void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max);

	/// Write properties of element to ostream
	/** Used in Field.serialize(os) to provide summary of field.
	  */ 
	friend std::ostream& operator<<(
		std::ostream&                     os,
		const ElementIsotropic<dim,real>& element) 
	{
		os << "Isotropic element with properties:" << std::endl
		   << '\t' << "m_scale = " << element.m_scale << std::endl;

		return os;
	}

private:
	/// element size
	real m_scale;
};

/// Anisotropic element class
/** Specialization of the element type for the anisotropic remeshing case.
  * Stores decomposed frame field axes (size, orientation and anisotropy).
  * A collection of elements is contained in the corresponding Field class.
  */
template <int dim, typename real>
class ElementAnisotropic : public Element<dim,real>
{
public:
	/// Constructor, sets default element definition
	/** Sets the scale to 0 (non-existant) and axes
	  * to the unit reference coordinate axes.
	  */
	ElementAnisotropic();

	/// Reference for element size
	/** Allows direct read/write of scale of mean element axis.
	  * Measure of element (length, area, volume) will be \f$scale^{dim}\f$
	  */
	real& scale() override;

	/// Set the scale for the element
	void set_scale(
		const real val) override;

	/// Get the scale for the element
	real get_scale() const override;

	/// Set the anisotropic ratio for each reference axis
	/** Requires array of order matching axis definition. 
	  * Each reference axis will have length \f$l = \alpha * scale\f$.
	  * Note: does nothing in the isotropic case. 
	  */
	void set_anisotropic_ratio(
		const std::array<real,dim>& ratio) override;

	/// Get the anisotropic ratio of each reference axis as an array
	/** Note: equals 1 for each axis in isotropic case.
	  */ 
	std::array<real,dim> get_anisotropic_ratio() override;

	/// Get the anisotropic ratio corresponding to the \f$j^{th}\f$ reference axis
	/** Note: equals 1 for each axis in isotropic case.
	  */ 
	real get_anisotropic_ratio(
		const unsigned int j) override;

	/// setting the (unit) axis direction
	void set_unit_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) override;

	/// getting the (unit) axis direction (array)
	std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis() override;

	/// getting the (unit) axis direction
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int j) override;

	/// setting frame axis j (scaled) at index
	void set_axis(
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) override;

	/// getting frame axis j (scaled) at index (array)
	std::array<dealii::Tensor<1,dim,real>,dim> get_axis() override;

	/// getting frame axis j (scaled) at index
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int j) override;

	/// get metric value at index
	dealii::Tensor<2,dim,real> get_metric() override;

	/// get inverse metric value
	dealii::Tensor<2,dim,real> get_inverse_metric() override;

protected:
	/// Set element to match geometry of input vertex set
	/** Vertices describe the tensor product element in Deal.II ordering.
	  * Internal function used in handling of set_cell().
	  */
	void set_cell_internal(
		const typename Element<dim,real>::VertexList& vertices) override;

public:
	/// Set anisotropy from reconstructed directional derivatives
	/** Based on Dolejsi's method for simplices, uses values obtained from
	  * reconstructed polynomial on neighbouring cell patch.
	  */
	void set_anisotropy(
		const std::array<real,dim>&                       derivative_value,
		const std::array<dealii::Tensor<1,dim,real>,dim>& derivative_direction,
		const unsigned int                                order) override;

	/// Limits the anisotropic ratios to a given bandwidth
	/** First finds ratio above max, redistributes length change to maintain constant volume and scale.
	  * Then the process is repeated with a lower bound.
	  * Note: does nothing in the isotropic case.
	  */ 
	void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max);

	/// Write properties of element to ostream
	/** Used in Field.serialize(os) to provide summary of field.
	  */ 
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

	/// resets the element
	void clear();

	/// Corrects internal values to ensure consistent treatement
	/** Element size, direction unit vectors and anisotropic ratios
	  * are renormalized based on expected definitions.
	  */
	void correct_element();

	/// renormalize the unit_axis
	/** Uses factoring and rescaling non-unit length into anisotropic ratio.
	  * correct_anisotropic_ratio() should be called immediately afterward.
	  */
	void correct_unit_axis();

	/// Correct the anisotropic ratio values
	/** Guarantees that their product is equal to 1
	  */ 
	void correct_anisotropic_ratio();

	/// Sorts the anisotropic ratio values (and corresponding unit axis) in ascending order
	void sort_anisotropic_ratio();

	/// element size based on \f$(Volume)^{1/dim}\f$
	real m_scale;

	/// Anisotropic axis ratios (relative to element scale)
	std::array<real, dim> m_anisotropic_ratio;

	/// axis unit vector directions
	std::array<dealii::Tensor<1,dim,real>, dim> m_unit_axis;
};

/// Field element class
/** This class describes the anisotropic size specification for continuous
  * mesh remeshing methods. Each DG mesh element is associated with an isotropic
  * or anisotropic description for the ideal target element at each point during remeshing.
  * Provides wraper to access these values to output to mesh generators without knowledge of mesh type. 
  * Each local element is stored in ElementIsotropic or ELementAnisotropic
  * specified above and the FieldInternal class is used to differentiate.
  */
template <int dim, typename real>
class Field 
{
public:
	/// reinitialize the internal data structure 
	virtual void reinit(
		const unsigned int size) = 0;

	/// returns the internal vector size
	virtual unsigned int size() const = 0;

	/// reference for element size
	virtual real& scale(
		const unsigned int index) = 0;

	/// Sets scale for the specified element index
	virtual void set_scale(
		const unsigned int index,
		const real         val) = 0;

	/// Gets scale for the specified element index
	virtual real get_scale(
		const unsigned int index) const = 0;

	/// Sets scale for all elements from a scale vector (dealii::Vector)
	void set_scale_vector_dealii(
		const dealii::Vector<real>& vec);
	
	/// Sets scale for all elements from a scale vector(std::vector)
	void set_scale_vector(
		const std::vector<real>& vec);

	/// Gets scale for all elements from a scale vector (dealii::Vector)
	dealii::Vector<real> get_scale_vector_dealii() const;

	/// Gets scale fpr all elements from a scale vector (std::vector)
	std::vector<real> get_scale_vector() const;

	/// Sets anisotropic ratio for the specified element index (if anisotropic)
	virtual void set_anisotropic_ratio(
		const unsigned int          index,
		const std::array<real,dim>& ratio) = 0;

	/// Gets the anisotropic ratio for the specified element index
	virtual std::array<real,dim> get_anisotropic_ratio(
		const unsigned int index) = 0;

	/// Gets the anisotropic ratio for the specified element index along axis j
	virtual real get_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j) = 0;

	/// Gets the vector of largest anisotropic ratio for each element (dealii::Vector)
	dealii::Vector<real> get_max_anisotropic_ratio_vector_dealii();

	/// Sets the group of dim (unit) axis directions for the specified element index
	virtual void set_unit_axis(
		const unsigned int                                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) = 0;

	/// Gets the group of dim (unit) axis direction for the specified element index
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis(
		const unsigned int index) = 0;

	/// Sets the j^th (unit) axis direction for the specified element index
	virtual dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int index,
		const unsigned int j) = 0;

	/// Sets the j^th (scale) axis direction for the specified element index
	virtual void set_axis(
		const unsigned int                                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) = 0;

	/// Gets the frame (dim scaled axis vectors) for the specified element index
	virtual std::array<dealii::Tensor<1,dim,real>,dim> get_axis(
		const unsigned int index) = 0;

	/// Gets the j^th frame component (scaled axis vector) for the specified element index
	virtual dealii::Tensor<1,dim,real> get_axis(
		const unsigned int index,
		const unsigned int j) = 0;

	/// Sets the frame (dim scaled axis vectors) for each element (std::vector)
	void set_axis_vector(
		const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& vec);

	/// Gets the frame (dim scaled axis vectors) for each element (std::vector)
	std::vector<std::array<dealii::Tensor<1,dim,real>,dim>> get_axis_vector();

	/// Sets the j^th frame component (scaled axis vector) for each element (std::vector)
	std::vector<dealii::Tensor<1,dim,real>> get_axis_vector(
		const unsigned int j);

	/// Gets the anisotropic metric tensor, \f$M\f$, for the specified element index
	virtual dealii::Tensor<2,dim,real> get_metric(
		const unsigned int index) = 0;

	/// Gets the anisotropic metric tensor, \f$M\f$, for each element (std::vector)
	std::vector<dealii::Tensor<2,dim,real>> get_metric_vector();

	/// Gets the inverse metric tensor, \f$M^{-1}\f$, for the specified element index
	virtual dealii::Tensor<2,dim,real> get_inverse_metric(
		const unsigned int index) = 0;

	/// Gets the inverse metric tensor, \f$M^{-1}\f$,  for each element (std::vector)
	std::vector<dealii::Tensor<2,dim,real>> get_inverse_metric_vector();

	/// Gets the quadratic Riemannian metric, \f$\mathcal{M} = M^T M\f$, for the specified element index
	dealii::SymmetricTensor<2,dim,real> get_quadratic_metric(
		const unsigned int index);

	/// Gets the quadratic Riemannian metric, \f$\mathcal{M} = M^T M\f$, for each element (std::vector)
	std::vector<dealii::SymmetricTensor<2,dim,real>> get_quadratic_metric_vector();

	/// Gets the inverse quadratic Riemannian metric used with BAMG, \f$\mathcal{M}^{-1}\f$, for a specified element index
	dealii::SymmetricTensor<2,dim,real> get_inverse_quadratic_metric(
		const unsigned int index);

	/// Gets the inverse quadratic Riemannian metric used with BAMG, \f$\mathcal{M}^{-1}\f$, for each element (std::vector)
	std::vector<dealii::SymmetricTensor<2,dim,real>> get_inverse_quadratic_metric_vector();

	/// Associated DofHandler type
	using DoFHandlerType = dealii::DoFHandler<dim>;

	/// Assigns the existing field based on an input DoFHandlerType
	virtual void set_cell(
		const DoFHandlerType& dof_handler) = 0;

	/// Compute anisotropic ratio from directional derivatives
	/** Uses Dolejsi's anisotropy method based on reconstructed \f$p+1\f$ directional derivatives.
	  * Derivatives are obtained in reconstruct_poly.cpp.
	  */ 
	virtual void set_anisotropy(
		const dealii::DoFHandler<dim>&                                 dof_handler,
		const std::vector<std::array<real,dim>>&                       derivative_value,
		const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& derivative_direction,
		const int                                                      relative_order) = 0;
	
	/// Globally limit anisotropic ratio to a specified range
	virtual void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max) = 0;

	/// Performs internal call for writing the field description to an ostream
	virtual std::ostream& serialize(
		std::ostream& os) const = 0;

	/// Performs output to ostream using internal serialization
	friend std::ostream& operator<<(
		std::ostream&          os,
		const Field<dim,real>& field)
	{
		return field.serialize(os);
	}

};

/// Internal Field element class
/** This class reimplements the above generalized function with internal handling of the 
  * ElementType. This prevents i/o and certain adaption functions from needing to directly
  * check the type of field which is being used. Internally stores the target output mesh 
  * description from the continuous space as a discrete vector associated with each element.
  */
template <int dim, typename real, typename ElementType>
class FieldInternal : public Field<dim,real>
{
public:
	/// reinitialize the internal data structure 
	
	void reinit(
		const unsigned int size);

	/// returns the internal vector size
	unsigned int size() const;

	/// reference for element size of specified element index
	real& scale(
		const unsigned int index) override;

	/// Sets scale for the specified element index
	void set_scale(
		const unsigned int index,
		const real         val) override;

	/// Gets scale for the specified element index
	real get_scale(
		const unsigned int index) const override;

	/// Sets anisotropic ratio for the specified element index (if anisotropic)
	void set_anisotropic_ratio(
		const unsigned int          index,
		const std::array<real,dim>& ratio) override;

	/// Gets the anisotropic ratio for the specified element index
	std::array<real,dim> get_anisotropic_ratio(
		const unsigned int index) override;

	/// Gets the anisotropic ratio for the specified element index along axis j
	real get_anisotropic_ratio(
		const unsigned int index,
		const unsigned int j) override;

	/// Sets the group of dim (unit) axis directions for the specified element index
	void set_unit_axis(
		const unsigned int                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& unit_axis) override;

	/// Sets the group of dim (unit) axis directions for the specified element index
	std::array<dealii::Tensor<1,dim,real>,dim> get_unit_axis(
		const unsigned int index) override;

	/// Gets the group of dim (unit) axis direction for the specified element index
	dealii::Tensor<1,dim,real> get_unit_axis(
		const unsigned int index,
		const unsigned int j) override;

	/// Sets the j^th (unit) axis direction for the specified element index
	void set_axis(
		const unsigned int                                index,
		const std::array<dealii::Tensor<1,dim,real>,dim>& axis) override;

	/// Gets the frame (dim scaled axis vectors) for the specified element index
	std::array<dealii::Tensor<1,dim,real>,dim> get_axis(
		const unsigned int index) override;

	/// Gets the j^th frame component (scaled axis vector) for the specified element index
	dealii::Tensor<1,dim,real> get_axis(
		const unsigned int index,
		const unsigned int j) override;

	// Gets the anisotropic metric tensor, \f$M\f$, for the specified element index
	dealii::Tensor<2,dim,real> get_metric(
		const unsigned int index) override;

	//  Gets the inverse metric tensor, \f$M^{-1}\f$, for the specified element index
	dealii::Tensor<2,dim,real> get_inverse_metric(
		const unsigned int index) override;

	// Assigns the existing field based on an input DoFHandlerType
	void set_cell(
		const typename Field<dim,real>::DoFHandlerType& dof_handler) override
	{
		reinit(dof_handler.get_triangulation().n_active_cells());

		for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
			if(cell->is_locally_owned())
				field[cell->active_cell_index()].set_cell(cell);
	}

	/// Compute anisotropic ratio from directional derivatives
	/** Uses Dolejsi's anisotropy method based on reconstructed \f$p+1\f$ directional derivatives.
	  * Derivatives are obtained in reconstruct_poly.cpp.
	  */ 
	void set_anisotropy(
		const dealii::DoFHandler<dim>&                                 dof_handler,
		const std::vector<std::array<real,dim>>&                       derivative_value,
		const std::vector<std::array<dealii::Tensor<1,dim,real>,dim>>& derivative_direction,
		const int                                                      relative_order) override;

	/// Globally limit anisotropic ratio to a specified range
	void apply_anisotropic_limit(
		const real anisotropic_ratio_min,
		const real anisotropic_ratio_max) override;

	/// Performs internal call for writing the field description to an ostream
	std::ostream& serialize(
		std::ostream& os) const override;

private:
	/// Internal vector storage of element data for each index
	std::vector<ElementType> field;
};

/// Field with isotropic element 
/** Describes target size field for the mesh only
  */ 
template <int dim, typename real>
using FieldIsotropic = FieldInternal<dim,real,ElementIsotropic<dim,real>>;

/// Field with anisotropic element
/** Describes mesh size, orientation and anisotropy. Description is based
  * on storage of frame axes for the target tensor-product element in dim dimensions.
  */
template <int dim, typename real>
using FieldAnisotropic = FieldInternal<dim,real,ElementAnisotropic<dim,real>>;

} // namespace GridRefinement

} // namespace PHiLiP

#endif //__FIELD_H__
