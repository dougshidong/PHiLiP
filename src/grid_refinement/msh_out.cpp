#include <float.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

#include "msh_out.h"

namespace PHiLiP {

namespace GridRefinement {

/* set of functions for outputting the mesh and data in .msh v4.1 format. 
 * contains 3 main sections:
 *      $MeshFormat - file description
 *      $Nodes      - physical points of the mesh
 *      $Elements   - element connectivity of mesh
 * 
 * Additional optional section (comes before $Nodes):
 *      $Entities - geometric description of the domain
 * 
 * And for data handling, may include multiple of section types:
 *      $NodeData        - data field stored nodewise
 *      $ElementData     - data field stored elementwise
 *      $ElementNodeData - data field stored nodes of each element (DG)
 * 
 * See following for details:
 *      https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
 */
// writing the mesh output with data if specified
template <int dim, typename real>
void MshOut<dim,real>::write_msh(
    std::ostream &out)
{
    // $MeshFormat
    out << "$MeshFormat" << '\n';

    // version(ASCII double; currently 4.1) file-type(ASCII int; 0 for ASCII mode, 1 for binary mode) data-size(ASCII int; sizeof(size_t))
    out << "4.1 0 8" << '\n';

    // $EndMeshFormat
    out << "$EndMeshFormat" << '\n';

    // $Entities
    // NOT IMPLEMENTED
    // $EndEntities

    // $Nodes
    out << "$Nodes" << '\n';

    const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
    
    const std::vector<dealii::Point<dim>> &vertices = tria.get_vertices();

    const std::vector<bool> &vertex_used = tria.get_used_vertices();
    const unsigned int       n_vertices  = tria.n_used_vertices();

    // numEntityBlocks(size_t) numNodes(size_t) minNodeTag(size_t) maxNodeTag(size_t)
    out << 1 << " " << n_vertices << " " << 1 << " " << vertices.size() << '\n';

    // entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesInBlock(size_t)
    out << dim << " " << 1 << " " << 0 << " " << n_vertices << '\n';
    
    // looping over the nodes of the triangulation
    // nodeTag(size_t)
    for(unsigned int i = 0; i < vertices.size(); ++i){
        if(!vertex_used[i]) continue;

        unsigned int node_tag = i + 1;
        out << node_tag << '\n';
    }

    // x(double) y(double) z(double)
    for(unsigned int i = 0; i < vertices.size(); ++i){
        if(!vertex_used[i]) continue;

        out << vertices[i];
        for(unsigned int d = dim; d < 3; ++d)
            out << " " << 0;

        out << '\n';
    }

    // $EndNodes
    out << "$EndNodes" << '\n';


    // $Elements
    out << "$Elements" << '\n';

    // elementType:
    // 1 - 2 node line
    // 3 - 4 node quadrangle
    // 5 - 8 node hexahedron
    unsigned int element_type;
    switch(dim){
        case 1: 
            element_type = 1;
            break;

        case 2:
            element_type = 3;
            break;

        case 3: 
            element_type = 5;
            break;
    }


    // numEntityBlocks(size_t) numElements(size_t) minElementTag(size_t) maxElementTag(size_t)
    out << 1 << " " << tria.n_active_cells() << " " << 1 << " " << tria.n_cells() << '\n';

    // entityDim(int) entityTag(int) elementType(int; see above) numElementsInBlock(size_t)
    out << dim << " " << 1 << " " << element_type << " " << tria.n_active_cells() << '\n';

    // elementTag(size_t) nodeTag(size_t) ...
    for(auto cell = tria.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int element_tag = cell->active_cell_index() + 1;
        out << element_tag;

        // switching numbering order to match mesh writing, nodeTag = nodeIndex + 1
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
            unsigned int node_tag = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]) + 1;
            out << " " << node_tag;
        }
        
        out << '\n';
    }

    // $EndElements
    out << "$EndElements" << '\n';

    // data sections
    for(auto gmsh_data: data_vector)
        gmsh_data->write_msh_data(dof_handler, out);

    out << std::flush;
}

// writing the data from a MshOutData
template <int dim>
void MshOutData<dim>::write_msh_data(
    const dealii::DoFHandler<dim> &dof_handler,
    std::ostream &                 out)
{
    // opening section
    switch(storage_type){
        case StorageType::node:
            out << "$NodeData" << '\n';
            break;

        case StorageType::element:
            out << "$ElementData" << '\n';
            break;
            
        case StorageType::elementNode:
            out << "$ElementNodeData" << '\n';
            break;
    }

    // writing the section header
    int num_string_tags  = string_tags.size(); 
    int num_real_tags    = real_tags.size();
    int num_integer_tags = integer_tags.size();

    // numStringTags(ASCII int)
    // stringTag(string) ...
    out << num_string_tags << '\n';
    for(auto string_tag: string_tags)
        out << '"' << string_tag << '"' << '\n';

    // numRealTags(ASCII int)
    // realTag(ASCII double) ...
    out << num_real_tags << '\n';
    for(auto real_tag: real_tags)
        out << real_tag << '\n';

    // numIntegerTags(ASCII int)
    // integerTag(ASCII int) ...
    out << num_integer_tags << '\n';
    for(auto integer_tag: integer_tags)
        out << integer_tag << '\n';

    // writing the data (internal)
    write_msh_data_internal(dof_handler, out);

    // closing the section
    switch(storage_type){
        case StorageType::node:
            out << "$EndNodeData" << '\n';
            break;

        case StorageType::element:
            out << "$EndElementData" << '\n';
            break;

        case StorageType::elementNode:
            out << "$EndElementNodeData" << '\n';
            break;
    }
}

// getting the number of entries (rows associated with the section)
template <int dim>
unsigned int MshOutData<dim>::num_entries(
    const dealii::DoFHandler<dim> &dof_handler)
{
    switch(storage_type){
        case StorageType::node:
            return dof_handler.get_triangulation().n_used_vertices();

        case StorageType::element:
        case StorageType::elementNode:
            return dof_handler.get_triangulation().n_active_cells();
    }

    return 0;
}

// sets the string tags
template <int dim>
void MshOutData<dim>::set_string_tags(
    std::string name,
    std::string interpolation_scheme)
{
    string_tags.push_back(name);
    string_tags.push_back(interpolation_scheme);
}

// sets only the name
template <int dim>
void MshOutData<dim>::set_string_tags(
    std::string name)
{
    string_tags.push_back(name);
}

// sets the real tags
template <int dim>
void MshOutData<dim>::set_real_tags(
    double time)
{
    real_tags.push_back(time);
}

// sets the integer tags
template <int dim>
void MshOutData<dim>::set_integer_tags(
    unsigned int time_step,
    unsigned int num_components,
    unsigned int num_entries)
{
    integer_tags.push_back(time_step);
    integer_tags.push_back(num_components);
    integer_tags.push_back(num_entries);
}

// different data output types
using Scalar = double;
using Vector = dealii::Tensor<1,PHILIP_DIM,double>;
using Matrix = dealii::Tensor<2,PHILIP_DIM,double>;

// specifying the data size of each entry for storage
template <>
unsigned int const MshOutDataInternal<PHILIP_DIM,Scalar>::num_components = 1;

template <>
unsigned int const MshOutDataInternal<PHILIP_DIM,Vector>::num_components = 3;

template <>
unsigned int const MshOutDataInternal<PHILIP_DIM,Matrix>::num_components = 9;

// writing the data for scalar data
template <>
void MshOutDataInternal<PHILIP_DIM,Scalar>::write_msh_data_internal(
    const dealii::DoFHandler<PHILIP_DIM> &dof_handler,
    std::ostream &                        out)
{
    const unsigned int dim = PHILIP_DIM;

    // looping through the data structure
    switch(storage_type){
        case StorageType::node:{
            // nodeTag(size_t) value(double) ...
            const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
            
            const std::vector<bool> &vertex_used = tria.get_used_vertices();

            // looping over the nodes of the triangulation
            for(unsigned int i = 0; i < vertex_used.size(); ++i){
                if(!vertex_used[i]) continue;

                Scalar node_data = data[i];

                // data entries
                unsigned int node_tag = i + 1;
                out << node_tag << " " << node_data << '\n';
            }

            break;

        }case StorageType::element:{
            // elementTag(size_t) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                Scalar cell_data = data[cell->active_cell_index()];

                // data entries
                unsigned int element_tag =  cell->active_cell_index() + 1;
                out << element_tag << " " << cell_data << '\n';
            }

            break;

        }case StorageType::elementNode:{
            // elementTag(size_t) numNodesPerElement(int) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                unsigned int element_tag = cell->active_cell_index() + 1;
                out << element_tag << " " << dealii::GeometryInfo<dim>::vertices_per_cell;

                // looping over the vertices
                for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
                    // nodeTag = nodeIndex + 1
                    unsigned int node_index = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]);
                    
                    Scalar node_data = data[node_index];

                    out << " " << node_data;
                }

                out << '\n';
            }

            break;
        }
    }
}

// writing the data for vector data
template <>
void MshOutDataInternal<PHILIP_DIM,Vector>::write_msh_data_internal(
    const dealii::DoFHandler<PHILIP_DIM> &dof_handler,
    std::ostream &                        out)
{
    const unsigned int dim = PHILIP_DIM;

    // looping through the data structure
    switch(storage_type){
        case StorageType::node:{
            // nodeTag(size_t) value(double) ...
            const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
            
            const std::vector<bool> &vertex_used = tria.get_used_vertices();

            // looping over the nodes of the triangulation
            for(unsigned int i = 0; i < vertex_used.size(); ++i){
                if(!vertex_used[i]) continue;

                Vector node_data = data[i];
                
                unsigned int node_tag = i + 1;
                out << node_tag;

                // vector entries
                for(unsigned int i = 0; i < dim; ++i)
                    out << " " << node_data[i];
                
                // padding zeros
                for(unsigned int i = dim; i < 3; ++i)
                    out << " " << 0;

                out << '\n';
            }

            break;

        }case StorageType::element:{
            // elementTag(size_t) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                Vector cell_data = data[cell->active_cell_index()];
                
                unsigned int element_tag = cell->active_cell_index() + 1;
                out << element_tag;

                // vector entries
                for(unsigned int i = 0; i < dim; ++i)
                    out << " " << cell_data[i];

                // padding zeros
                for(unsigned int i = dim; i < 3; ++i)
                    out << " " << 0;

                out << '\n';
            }

            break;

        }case StorageType::elementNode:{
            // elementTag(size_t) numNodesPerElement(int) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                unsigned int element_tag = cell->active_cell_index() + 1;
                out << element_tag << " " << dealii::GeometryInfo<dim>::vertices_per_cell;

                // looping over the vertices
                for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
                    // switching numbering order to match mesh writing, nodeTag = nodeIndex + 1
                    unsigned int node_index = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]);
                    
                    Vector node_data = data[node_index];

                    // vector entries
                    for(unsigned int i = 0; i < dim; ++i)
                        out << " " << node_data[i];

                    // padding zeros
                    for(unsigned int i = dim; i < 3; ++i)
                        out << " " << 0;
                        
                }

                out << '\n';
            }

            break;
        }
    }
}

// writing the data for matrix data
template <>
void MshOutDataInternal<PHILIP_DIM,Matrix>::write_msh_data_internal(
    const dealii::DoFHandler<PHILIP_DIM> &dof_handler,
    std::ostream &                        out)
{
    const unsigned int dim = PHILIP_DIM;

    // looping through the data structure
    switch(storage_type){
        case StorageType::node:{
            // nodeTag(size_t) value(double) ...
            const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
            
            const std::vector<bool> &vertex_used = tria.get_used_vertices();

            // looping over the nodes of the triangulation
            for(unsigned int i = 0; i < vertex_used.size(); ++i){
                if(!vertex_used[i]) continue;

                Matrix node_data = data[i];
                
                unsigned int node_tag = i + 1;
                out << node_tag;

                for(unsigned int i = 0; i < dim; ++i){
                    // matrix entries
                    for(unsigned int j = 0; j < dim; ++j)
                        out << " " << node_data[i][j];
                    
                    // padding zeros
                    for(unsigned int j = dim; j < 3; ++j)
                        out << " " << 0;
                }

                // padding zeros
                for(unsigned int i = dim; i < 3; ++i)
                    for(unsigned int j = 0; j < 3; ++j)
                        out << " " << 0;

                out << '\n';
            }

            break;

        }case StorageType::element:{
            // elementTag(size_t) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                Matrix cell_data = data[cell->active_cell_index()];
                
                unsigned int element_tag = cell->active_cell_index() + 1;
                out << element_tag;

                for(unsigned int i = 0; i < dim; ++i){
                    // matrix entries
                    for(unsigned int j = 0; j < dim; ++j)
                        out << " " << cell_data[i][j];
                    
                    // padding zeros
                    for(unsigned int j = dim; j < 3; ++j)
                        out << " " << 0;
                }

                // padding zeros
                for(unsigned int i = dim; i < 3; ++i)
                    for(unsigned int j = 0; j < 3; ++j)
                        out << " " << 0;

                out << '\n';
            }

            break;

        }case StorageType::elementNode:{
            // elementTag(size_t) numNodesPerElement(int) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                unsigned int elementTag = cell->active_cell_index() + 1;
                out << elementTag << " " << dealii::GeometryInfo<dim>::vertices_per_cell;

                // looping over the vertices
                for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
                    // switching numbering order to match mesh writing, nodeTag = nodeIndex + 1
                    unsigned int node_index = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]);
                    
                    Matrix node_data = data[node_index];

                    for(unsigned int i = 0; i < dim; ++i){
                        // matrix entries
                        for(unsigned int j = 0; j < dim; ++j)
                            out << " " << node_data[i][j];
                        
                        // padding zeros
                        for(unsigned int j = dim; j < 3; ++j)
                            out << " " << 0;
                    }

                    // padding zeros
                    for(unsigned int i = dim; i < 3; ++i)
                        for(unsigned int j = 0; j < 3; ++j)
                            out << " " << 0;
                            
                }

                out << '\n';
            }

            break;
        }
    }
}

template class MshOut <PHILIP_DIM, double>;
template class MshOutData <PHILIP_DIM>;

} // namespace GridRefinement

} //namespace PHiLiP