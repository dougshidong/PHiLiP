#ifndef __MSH_OUT_H__
#define __MSH_OUT_H__

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

namespace PHiLiP {

namespace GridRefinement {

// enum of mesh storage types
enum StorageType {
    node,
    element,
    elementNode,
};

// data structure for outputting data to msh files
template <int dim>
class MshOutData
{
public:
    // constructor
    MshOutData(
        StorageType storage_type) :
            storage_type(storage_type){};

    void write_msh_data(
        const dealii::hp::DoFHandler<dim> &dof_handler,
        std::ostream &                     out);

protected:
    // storage type of the current object
    StorageType storage_type;

    std::vector<std::string> string_tags;
    std::vector<double>      real_tags;
    std::vector<int>         integer_tags;

    // write function for the data to be overriden
    virtual void write_msh_data_internal(
        const dealii::hp::DoFHandler<dim> &dof_handler,
        std::ostream &                     out) = 0;

    // gets the number of entries
    unsigned int num_entries(
        const dealii::hp::DoFHandler<dim> &dof_handler);

    // sets the string tags
    void set_string_tags(
        std::string name,
        std::string interpolation_scheme);

    // sets the real tags
    void set_real_tags(
        double time);

    // sets the integer tags
    void set_integer_tags(
        unsigned int time_step,
        unsigned int num_components,
        unsigned int num_entries);
};

// templated class for internal data processing
template <int dim, typename T>
class MshOutDataInternal : public MshOutData<dim>
{
public:
    // constructor
    MshOutDataInternal(
        std::vector<T> data,
        StorageType    storage_type,
        const dealii::hp::DoFHandler<dim> &dof_handler) : 
            MshOutData<dim>(storage_type),
            data(data)
    {
        this->set_integer_tags(0, num_components, this->num_entries(dof_handler));
    };

protected:
    void write_msh_data_internal(
        const dealii::hp::DoFHandler<dim> &dof_handler,
        std::ostream &                     out) override;

private:
    const std::vector<T> data;

    // specifies the number of entries per data point
    static const unsigned int num_components;
};

// output class for Gmsh MSH v4.1 format
// extends capabilities of built in DealII data structure to handle data output
template <int dim, typename real>
class MshOut
{
public: 
    // constructor
    MshOut(
        const dealii::hp::DoFHandler<dim> &dof_handler) :
            dof_handler(dof_handler){};

    // adding data vector of specified storage type
    template <typename T>
    void add_data_vector(
        std::vector<T> data,
        StorageType    storage_type)
    {
        data_vector.push_back(std::make_shared<MshOutDataInternal<dim,T>>(data,storage_type,dof_handler));
    }

    // performing the output to ostream
    void write_msh(
        std::ostream &out);

private:
        const dealii::hp::DoFHandler<dim> &           dof_handler;
        std::vector<std::shared_ptr<MshOutData<dim>>> data_vector;
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GMSH_OUT_H__

