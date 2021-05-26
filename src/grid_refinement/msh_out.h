#ifndef __MSH_OUT_H__
#define __MSH_OUT_H__

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

namespace PHiLiP {

namespace GridRefinement {

/// Enum of mesh storage locations for .msh data fields
enum class StorageType{
    node,        ///< Values stored at mesh nodes
    element,     ///< Values stored at mesh elements (\f$C^0\f$)
    elementNode, ///< Values stored at each node of each element (\f$C^1\f$)
};

/// Enum of data types for .msh data fields
enum class DataType{
    scalar, ///< Scalar data entries 
    vector, ///< Vector data entries (\f$3\times1\f$)
    matrix, ///< Matrix data entries (\f$3\times3\f$)
};

/// Data structure class for data fields of .msh files
/** Contains functionality to store and output mesh associated
  * data for .msh v4.1 file formats suitable for use with GMSH.
  * Also used to export metric data to external LpCVT mesh
  * generation program. Tags and other functionality included 
  * to distinguish between multiple field types. Internally handles
  * the different types data availible.
  */ 
template <int dim>
class MshOutData
{
public:
    /// Delegated constructor
    /** Sets only the internal storage type for header creation
      */ 
    MshOutData(
        StorageType storage_type) :
            storage_type(storage_type){};

    /// Perform writing of .msh file data section
    /** Writes header and associated information tags then
      * calls the internal function specific to the
      * data storage type and value type to output data.
      */
    void write_msh_data(
        const dealii::DoFHandler<dim> &dof_handler,
        std::ostream &                 out);

protected:
    /// Storage location of the .msh data field entries
    StorageType storage_type;

    /// String tags for the data field
    /** Should contain the name of the data field. Optionally the interpolation scheme
      * for data visualization can also be specified through this field. Other custom 
      * entries may also be added through the set_string_tags function.
      */ 
    std::vector<std::string> string_tags;  

    /// Real tags for the data field
    /** No specific entries are required but may be used to pass additional information
      * about the data field such as the time at current time step if multiple fields
      * are passed together. Setup using the set_real_tags function.
      */ 
    std::vector<double>      real_tags;    

    /// Integer tags for the data field.
    /** Should include the numbered time step, number of local data components (at each data point)
      * and the number of total entries (number of nodes or elements) where the data is written.
      * Setup using the set_integer_tags function.
      */
    std::vector<int>         integer_tags; 

    /// Perform write of internal data field body at storage location
    /** Function to be overwritten by the internal data structure type.
      * Format must be consistent with the num_entries and num_components
      * setup through the integer_tags field.
      */
    virtual void write_msh_data_internal(
        const dealii::DoFHandler<dim> &dof_handler,
        std::ostream &                 out) = 0;

    /// Gets the number of data entries associated with the mesh
    /** Can be the number of mesh nodes or elements depending 
      * on the storage location of the data entries.
      */ 
    unsigned int num_entries(
        const dealii::DoFHandler<dim> &dof_handler);

    /// Sets the string tags for the data field
    /** Default GMSH treatment for the data field name 
      * and optionally set interpolation scheme specification.
      */ 
    void set_string_tags(
        std::string name,
        std::string interpolation_scheme);

    /// Sets the string name tag for the data field
    /** Default internal treatement for data field name 
      */
    void set_string_tags(
        std::string name);

    /// Sets real tag to the data field
    /** Represents single double value for the field such as
      * solution time in unsteady results. 
      */ 
    void set_real_tags(
        double time);

    // sets the integer tags
    /// Sets the required integer tags for the data field
    /** Required values include the current solution time step,
      * num_components per entry
      */ 
    void set_integer_tags(
        unsigned int time_step,
        unsigned int num_components,
        unsigned int num_entries);
};

/// Internal data handler class for .msh file data fields
/** This clas is additionally templated on the data type
  * format used to store the internal values. Functionality
  * is indirectly templated on the DataType enum  to output 
  * each variety of nodewise, elementwise or hybrid data.
  */
template <int dim, typename T>
class MshOutDataInternal : public MshOutData<dim>
{
public:
    /// Construct data field
    /** Based on vector of data type, storage location for the specified
      * data and a mesh description (dealii::DoFHandler) to represent
      * the internal data field structure.
      */ 
    MshOutDataInternal(
        std::vector<T> data,
        StorageType    storage_type,
        const dealii::DoFHandler<dim> &dof_handler) : 
            MshOutData<dim>(storage_type),
            data(data)
    {
        this->set_integer_tags(0, num_components, this->num_entries(dof_handler));
    };

    /// Construct data field with name
    /** Based on vector of data type, storage location for the specified
      * data and a mesh description (dealii::DoFHandler) to represent
      * the internal data field structure. Additionally includes name tag 
      * to differentiate between fields.
      */ 
    MshOutDataInternal(
        std::vector<T>                 data,
        StorageType                    storage_type,
        std::string                    name,
        const dealii::DoFHandler<dim> &dof_handler) :
            MshOutData<dim>(storage_type),
            data(data)
    {
        this->set_integer_tags(0, num_components, this->num_entries(dof_handler));
        this->set_string_tags(name);
    }

protected:
    /// Perform write of internal data field body at storage location
    /** Function to be overwritten by the internal data structure type.
      * Format must be consistent with the num_entries and num_components
      * setup through the integer_tags field.
      */
    void write_msh_data_internal(
        const dealii::DoFHandler<dim> &dof_handler,
        std::ostream &                 out) override;

private:
    /// Internal data storage vector
    /** Special handling based on type in the msh_out.cpp file has been setup
      * for Scalar (double), Vector (dealii::Tensor<1,dim,double>) and Matrix
      * (dealii::Tensor<2,dim,double>) cases. Used to write entries in consistent
      * manner in the 3 dimensional augmented space (with zeroes added as needed).
      */ 
    const std::vector<T> data;

    /// The number of components per data point
    /** Based on data type. Regardless of mesh dimension these values are always 
      * stored in 3D space, therefore, 1 for scalar, 3 for vector, 9 for matrix.
      */
    static const unsigned int num_components;
};

/// Output class for GMSH .msh v4.1 file format
/** Contains functionality for writing the mesh connectivity and node position data as
  * well as keeping track of several storage fields (of potentionally different types and
  * storage locations). Class functions to write this collection data in standard form
  * allowing extended data types to be read in GMSH for visualization or as a method
  * of interfacing with the external LpCVT mesh generator using elementwise metric data.
  * Note: Currently this class only supports p1 all-quad mesh types with information
  *       accesed through a dealii::DoFHandler.
  */
template <int dim, typename real>
class MshOut
{
public: 
    /// Construct mesh output handler
    /** Mesh description dealii::DoFHandler used to specify the linear mesh to 
      * be written along with any additional data fields in a consistent way. 
      */ 
    MshOut(
        const dealii::DoFHandler<dim> &dof_handler) :
            dof_handler(dof_handler){};

    /// Add data vector of specified storage type and values
    /** Data can be stored at nodes, elements ($C^0$) or nodes of elements ($C^1$).
      * Internal data structure can take the form of a Scalar (double), Vector
      * (dealii::Tensor<1,dim,double>) or Matrix (dealii::Tensor<2,dim,double>)
      * which is augmented to be output in the 3D space along with the mesh.
      * Each additional data vector corresponds to an additional section in the
      * written .msh v4.1 output file as $NodeData, $ElementData or $ElementNodeData
      * respectively.
      */ 
    template <typename T>
    void add_data_vector(
        std::vector<T> data,
        StorageType    storage_type)
    {
        data_vector.push_back(
            std::make_shared<MshOutDataInternal<dim,T>>(
                data,
                storage_type,
                dof_handler));
    }

    /// Add data vector of specified storage type and values with name
    /** Data can be stored at nodes, elements ($C^0$) or nodes of elements ($C^1$).
      * Internal data structure can take the form of a Scalar (double), Vector
      * (dealii::Tensor<1,dim,double>) or Matrix (dealii::Tensor<2,dim,double>)
      * which is augmented to be output in the 3D space along with the mesh.
      * Each additional data vector corresponds to an additional section in the
      * written .msh v4.1 output file as $NodeData, $ElementData or $ElementNodeData
      * respectively. Name can be used to differentiate between data fields.
      */ 
    template <typename T>
    void add_data_vector(
        std::vector<T> data,
        StorageType    storage_type,
        std::string    name)
    {
        data_vector.push_back(
            std::make_shared<MshOutDataInternal<dim,T>>(
                data,
                storage_type,
                name,
                dof_handler));
    }

    /// Output formatted .msh v4.1 file with mesh description and data field
    /** Formatted based on the standard GMSH .msh v4.1 file format. Basic mesh description
      * is written in $MeshFormat (description), $Nodes (mesh points), $Elements (connectivity).
      * An additional data field is written for each added data_vector entry in the corresponding
      * storage type sections: $NodeData (data stored nodewise), $ElementData (data stored elementwise)
      * or $ElementNodeData (data stored at the nodes of each element). These can take the form of 
      * Scalar, Vector or Matrix entries. Note: Currently the geometric description of the domain
      * and its boundaries included in the $Entities subsection has not been implemented. 
      */ 
    void write_msh(
        std::ostream &out);

private:
    const dealii::DoFHandler<dim> &               dof_handler; ///< Mesh description for acccess to node location and connecitviity information
    std::vector<std::shared_ptr<MshOutData<dim>>> data_vector; ///< Vector of data field entries stored in MshOutDataInternal based on data type of entries
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GMSH_OUT_H__

