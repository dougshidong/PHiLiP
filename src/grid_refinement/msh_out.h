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
        StorageType storageType) :
            storageType(storageType){};

    void write_msh_data(
        const dealii::hp::DoFHandler<dim> &dof_handler,
        std::ostream &                     out);

protected:
    // storage type of the current object
    StorageType storageType;

    std::vector<std::string> stringTags;
    std::vector<double>      realTags;
    std::vector<int>         integerTags;

    // write function for the data to be overriden
    virtual void write_msh_data_internal(
        const dealii::hp::DoFHandler<dim> &dof_handler,
        std::ostream &                     out) = 0;
};

// templated class for internal data processing
template <int dim, typename T>
class MshOutDataInternal : public MshOutData<dim>
{
public:
    // constructor
    MshOutDataInternal(
        std::vector<T> data,
        StorageType    storageType) : 
            MshOutData<dim>(storageType),
            data(data){};

protected:
    void write_msh_data_internal(
        const dealii::hp::DoFHandler<dim> &dof_handler,
        std::ostream &                     out) override;

private:
    const std::vector<T> data;
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
        StorageType    storageType)
    {
        data_vector.push_back(std::make_shared<MshOutDataInternal<dim,T>>(data,storageType));
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

