
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>

#include <fstream>
#include <iostream> 

#include "dg/dg.h"
#include "parameters/parameters.h"
#include "physics/physics.h"
#include "grid_refinement/msh_out.h"

//  data types and storage types imported from msh_out.cpp
using DataType    = PHiLiP::GridRefinement::DataType;
using StorageType = PHiLiP::GridRefinement::StorageType;

// typename for each datatype
using Scalar = double;
using Vector = dealii::Tensor<1,PHILIP_DIM,double>;
using Matrix = dealii::Tensor<2,PHILIP_DIM,double>;

// functions for getting data values
template <typename T>
T data_function(dealii::Point<PHILIP_DIM,double> x);

// specializations
template <>
Scalar data_function<Scalar>(
    dealii::Point<PHILIP_DIM,double> x)
{
    const int dim = PHILIP_DIM;
    const int exp = 3;

    // constructing homogoneous power series of order exp
    // x^3 + y^3 + ... + x^2 y + ...
    Scalar val = 0;
    switch(dim){
        case 1:
            val += pow(x[0],exp);

            break;
        
        case 2:
            for(unsigned int i = 0; i <= exp; ++i){
                unsigned int j = exp-i;
                val += pow(x[0],i) * pow(x[1],j);
            }

            break;

        case 3:
            for(unsigned int i = 0; i <= exp; ++i)
                for(unsigned int j = 0; i+j <= exp; ++j){
                    unsigned int k = exp-i-j;

                    val += pow(x[0],i) * pow(x[1],j) * pow(x[2],k);
                }

            break;
    }

    return val;
}

template <>
Vector data_function<Vector>(
    dealii::Point<PHILIP_DIM,double> x)
{
    const int dim = PHILIP_DIM;
    const int exp = 3;

    // constructing gradient of homogenous power series of order exp
    // = {3x^2 + 2xy + ..., 3y^2 + ..., ...}
    Vector val;
    switch(dim){
        case 1:
            if(exp >= 1) val[0] += exp* pow(x[0],exp-1);

            break;

        case 2:
            for(unsigned int i = 0; i <= exp; ++i){
                unsigned int j = exp-i;

                if(i >= 1) val[0] += i * pow(x[0],i-1) * pow(x[1],j);
                if(j >= 1) val[1] += j * pow(x[0],i)   * pow(x[1],j-1); 
            }

            break;

        case 3:
            for(unsigned int i = 0; i <= exp; ++i)
                for(unsigned int j = 0; i+j <= exp; ++j){
                    unsigned int k = exp-i-j;
                    
                    if(i >= 1) val[0] += i * pow(x[0],i-1) * pow(x[1],j)   * pow(x[2],k);
                    if(j >= 1) val[1] += j * pow(x[0],i)   * pow(x[1],j-1) * pow(x[2],k);
                    if(k >= 1) val[2] += k * pow(x[0],i)   * pow(x[1],j)   * pow(x[2],k-1);
                }

            break;
    }

    return val;
}

template <>
Matrix data_function<Matrix>(
    dealii::Point<PHILIP_DIM,double> x)
{
    const int dim = PHILIP_DIM;
    const int exp = 3;

    // constructing the hessian of homogenous power series of order exp
    // = {{6x + 2y + ..., 2x+2y+..., ...},{..., 6y+2x, ...},...
    Matrix val;
    switch(dim){
        case 1:
            if(exp >= 2) val[0][0] += exp * (exp - 1) * pow(x[0],exp-2);

            break;

        case 2:
            for(unsigned int i = 0; i <= exp; ++i){
                unsigned int j = exp-i;

                if(i >= 2) val[0][0] += i * (i-1) * pow(x[0],i-2) * pow(x[1],j);
                if(j >= 2) val[1][1] += j * (j-1) * pow(x[0],i)   * pow(x[1],j-2);

                if(i >= 1 && j >= 1) val[0][1] += i * j * pow(x[0],i-1) * pow(x[1],j-1);
            }

            val[1][0] = val[0][1];

            break;

        case 3:
            for(unsigned int i = 0; i <= exp; ++i)
                for(unsigned int j = 0; i+j <= exp; ++j){
                    unsigned int k = exp-i-j;

                    if(i >= 2) val[0][0] += i * (i-1) * pow(x[0],i-2) * pow(x[1],j)   * pow(x[2],k);
                    if(j >= 2) val[1][1] += j * (j-1) * pow(x[0],i)   * pow(x[1],j-2) * pow(x[2],k);
                    if(k >= 2) val[2][2] += k * (k-1) * pow(x[0],i)   * pow(x[1],j)   * pow(x[2],k-2);

                    if(i >= 1 && j >= 1) val[0][1] += i * j * pow(x[0],i-1) * pow(x[1],j-1) * pow(x[2],k);
                    if(i >= 1 && k >= 1) val[0][2] += i * k * pow(x[0],i-1) * pow(x[1],j)   * pow(x[2],k-1);
                    if(j >= 1 && k >= 1) val[1][2] += j * k * pow(x[0],i)   * pow(x[1],j-1) * pow(x[2],k-1);
                }

            val[1][0] = val[0][1];
            val[2][0] = val[0][2];
            val[2][1] = val[1][2];

            break;
    }

    return val;
}

// generates fe_collection for dof_handler initialization
template <int dim>
dealii::hp::FECollection<dim> get_fe_collection(
    const unsigned int max_degree,
    const int nstate,
    const bool use_collocated_nodes)
{
    dealii::hp::FECollection<dim> fe_coll;
    
    // collocated nodes repeat degree = 1
    unsigned int degree = use_collocated_nodes?1:0;
    const dealii::FE_DGQ<dim> fe_dg(degree);
    const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
    fe_coll.push_back(fe_system);

    // looping over remaining degrees
    for(unsigned int degree = 1; degree <= max_degree; ++degree){
        const dealii::FE_DGQ<dim> fe_dg(degree);
        const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
        fe_coll.push_back(fe_system);
    }

    // returning
    return fe_coll;
}

// gets the length of vector that needs to be alocated for a given storage type
template <int dim>
unsigned int get_vector_length(
    const dealii::DoFHandler<dim> &dof_handler,
    const StorageType             &storage_type)
{
    switch(storage_type){
        case StorageType::node:
            return dof_handler.get_triangulation().n_used_vertices();

        case StorageType::element:
            return dof_handler.get_triangulation().n_active_cells();

        case StorageType::elementNode:
            return dof_handler.get_triangulation().n_active_cells() * dealii::GeometryInfo<dim>::vertices_per_cell;
    
        default: std::terminate();
    }
}

// getting the coordinate associated with an index for a given storage type
template <int dim>
std::vector<dealii::Point<dim,double>> get_vector_coord(
    const dealii::DoFHandler<dim> &dof_handler,
    const StorageType             &storage_type)
{
    // if storageType is node, directly accesible
    if(storage_type == StorageType::node)
        return dof_handler.get_triangulation().get_vertices();

    // otherwise allocating the vector of points with empty points (0)
    unsigned int n_coord = get_vector_length(dof_handler, storage_type);
    std::vector<dealii::Point<dim,double>> vector_coord(n_coord);

    // looping over entries
    unsigned int vertices_per_cell = dealii::GeometryInfo<dim>::vertices_per_cell;
    switch(storage_type){
        case StorageType::node:{
            return dof_handler.get_triangulation().get_vertices();

        }case StorageType::element:{
            // otherwise allocating the vector of points with empty points (0)
            unsigned int n_coord = get_vector_length(dof_handler, storage_type);
            std::vector<dealii::Point<dim,double>> vector_coord(n_coord);

            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                for(unsigned int vertex = 0; vertex < vertices_per_cell; ++vertex)
                    vector_coord[cell->active_cell_index()] += cell->vertex(vertex);
            
                vector_coord[cell->active_cell_index()] /= vertices_per_cell;
            }

            return vector_coord;

        }case StorageType::elementNode:{
            // otherwise allocating the vector of points with empty points (0)
            unsigned int n_coord = get_vector_length(dof_handler, storage_type);
            std::vector<dealii::Point<dim,double>> vector_coord(n_coord);

            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                for(unsigned int vertex = 0; vertex < vertices_per_cell; ++vertex)
                    vector_coord[cell->vertex_index(vertex)] = cell->vertex(vertex);
            }

            return vector_coord;
        }

        default: std::terminate();
    }

}

// helper to run a single case of output
template <int dim, typename T>
void msh_out_test_helper(
    const dealii::DoFHandler<dim> &dof_handler,
    const DataType                &data_type,
    const StorageType             &storage_type)
{
    // generating the MshOut
    PHiLiP::GridRefinement::MshOut<dim,double> msh_out(dof_handler);

    // getting the position of each node
    std::vector<dealii::Point<dim,double>> vector_coord = get_vector_coord(dof_handler, storage_type);

    // vector type is dependent on the data_type, evaluating from funcitons
    std::vector<T> vector_data(vector_coord.size());

    for(unsigned int i = 0; i < vector_coord.size(); ++i)
        vector_data[i] = data_function<T>(vector_coord[i]);

    msh_out.add_data_vector(vector_data, storage_type);

    // building the file name and output stream
    std::string write_msh_name =  dealii::Utilities::int_to_string(dim) + "d_test_msh";

    switch(data_type){
        case DataType::scalar:
            write_msh_name += "_scalar";
            break;

        case DataType::vector:
            write_msh_name += "_vector";
            break;

        case DataType::matrix:
            write_msh_name += "_matrix";
            break; 
    }

    switch(storage_type){
        case StorageType::node:
            write_msh_name += "_node";
            break;
            
        case StorageType::element:
            write_msh_name += "_element";
            break;

        case StorageType::elementNode:
            write_msh_name += "_elementNode";
            break;
    }

    write_msh_name += ".msh";
    std::ofstream out_msh(write_msh_name);

    // performing write to disk
    std::cout << "Writing \"" << write_msh_name << ".msh\"... ";
    msh_out.write_msh(out_msh);
    std::cout << "Done!" << std::endl;
}

// generating output file for each storageType and dataType using MshOut
int main(int argc, char *argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    
    const int dim = PHILIP_DIM;

    // grid parameters
    const unsigned int grid_size = 2;
    const double left  = 0.0;
    const double right = 1.0;

    // dof_handler parameters
    const unsigned int degree = 3;
    const unsigned int nstate = 1;
    const bool use_collocated_nodes = false;

    // generate grid
#if PHILIP_DIM==1
    dealii::Triangulation<dim> grid(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
    dealii::parallel::distributed::Triangulation<dim> grid(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

    dealii::GridGenerator::subdivided_hyper_cube(grid, grid_size, left, right);

    // initializing the dof_handler
    dealii::DoFHandler<dim> dof_handler(grid);
    dof_handler.initialize(grid, get_fe_collection<dim>(degree, nstate, use_collocated_nodes));

    // setting the cell fe_degree
    grid.prepare_coarsening_and_refinement();
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned()) cell->set_future_fe_index(degree);
    grid.execute_coarsening_and_refinement();

    // looping over the storage type and dataType
    std::vector<StorageType> storage_types {
        StorageType::node,
        StorageType::elementNode,
        StorageType::element
    };

    std::vector<DataType> data_types {
        DataType::scalar,
        DataType::vector,
        DataType::matrix
    };

    for(const auto &storage_type: storage_types){
        for(const auto &data_type: data_types){
            // calling appropriate helper function dependent on data_type
            // directs function call to scalar, vector, matrix functions
            switch(data_type){
                case DataType::scalar:
                    msh_out_test_helper<dim,Scalar>(dof_handler, data_type, storage_type);
                    break;

                case DataType::vector:
                    msh_out_test_helper<dim,Vector>(dof_handler, data_type, storage_type);
                    break;

                case DataType::matrix:
                    msh_out_test_helper<dim,Matrix>(dof_handler, data_type, storage_type);
                    break;
            }
        }
    }

}

