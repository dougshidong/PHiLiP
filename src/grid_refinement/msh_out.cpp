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
    out << 1 << " " << n_vertices << " " << 1 << " " << vertices.size() + 1 << '\n';

    // entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesInBlock(size_t)
    out << dim << " " << 1 << " " << 0 << " " << n_vertices << '\n';
    
    // looping over the nodes of the triangulation
    // nodeTag(size_t)
    for(unsigned int i = 0; i < vertices.size(); ++i){
        if(!vertex_used[i]) continue;

        unsigned int nodeTag = i + 1;
        out << nodeTag << '\n';
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
    out << 1 << " " << tria.n_active_cells() << " " << 1 << " " << tria.n_cells()  << '\n';

    // entityDim(int) entityTag(int) elementType(int; see above) numElementsInBlock(size_t)
    out << dim << " " << 1 << " " << element_type << " " << tria.n_active_cells() << '\n';

    // elementTag(size_t) nodeTag(size_t) ...
    for(auto cell = tria.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        out << cell->active_cell_index();

        // switching numbering order to match mesh writing, nodeTag = nodeIndex + 1
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
            out << ((vertex==0)?(""):(" ")) << cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]) + 1;
        
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
    const dealii::DoFHandler<dim, dim> &dof_handler,
    std::ostream &                      out)
{
    // opening section
    switch(storageType){
        case node:
            out << "$NodeData" << '\n';
            break;

        case element:
            out << "$ElementData" << '\n';
            break;
            
        case elementNode:
            out << "$ElementNodeData" << '\n';
            break;
    }

    // writing the section header
    int numStringTags  = stringTags.size(); 
    int numRealTags    = realTags.size();
    int numIntegerTags = integerTags.size();

    // numStringTags(ASCII int)
    // stringTag(string) ...
    out << numStringTags << '\n';
    for(auto stringTag: stringTags)
        out << stringTag << '\n';

    // numRealTags(ASCII int)
    // realTag(ASCII double) ...
    out << numRealTags << '\n';
    for(auto realTag: realTags)
        out << realTag << '\n';

    // numIntegerTags(ASCII int)
    // integerTag(ASCII int) ...
    out << numIntegerTags << '\n';
    for(auto integerTag: integerTags)
        out << integerTag << '\n';

    // writing the data (internal)
    write_msh_data_internal(dof_handler, out);

    // closing the section
    switch(storageType){
        case node:
            out << "$EndNodeData" << '\n';
            break;

        case element:
            out << "$EndElementData" << '\n';
            break;

        case elementNode:
            out << "$EndElementNodeData" << '\n';
            break;
    }
}

// writing the data for scalar data
template <>
void MshOutDataInternal<PHILIP_DIM,double>::write_msh_data_internal(
    const dealii::DoFHandler<PHILIP_DIM,PHILIP_DIM> &dof_handler,
    std::ostream &                                   out)
{
    const int dim = PHILIP_DIM;

    // looping through the data structure
    switch(storageType){
        case node:{
            // nodeTag(size_t) value(double) ...
            const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
            
            const std::vector<bool> &vertex_used = tria.get_used_vertices();

            // looping over the nodes of the triangulation
            for(unsigned int i = 0; i < vertex_used.size(); ++i){
                if(!vertex_used[i]) continue;

                double nodeData = data[i];

                // data entries
                unsigned int nodeTag = i + 1;
                out << nodeTag << " " << nodeData << '\n';
            }

            break;

        }case element:{
            // elementTag(size_t) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                double cellData = data[cell->active_cell_index()];

                // data entries
                unsigned int elementTag =  cell->active_cell_index() + 1;
                out << elementTag << " " << cellData << '\n';
            }

            break;

        }case elementNode:{
            // elementTag(size_t) numNodesPerElement(int) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                unsigned int elementTag = cell->active_cell_index() + 1;
                out << elementTag << " " << dealii::GeometryInfo<dim>::vertices_per_cell;

                // looping over the vertices
                for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
                    // nodeTag = nodeIndex + 1
                    unsigned int nodeIndex = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]);
                    
                    double nodeData = data[nodeIndex];

                    out << " " << nodeData;
                }

                out << '\n';
            }

            break;
        }
    }
}

// writing the data for vector data
template <>
void MshOutDataInternal<PHILIP_DIM,dealii::Tensor<1,PHILIP_DIM,double>>::write_msh_data_internal(
    const dealii::DoFHandler<PHILIP_DIM,PHILIP_DIM> &dof_handler,
    std::ostream &                                   out)
{
    const int dim = PHILIP_DIM;

    // looping through the data structure
    switch(storageType){
        case node:{
            // nodeTag(size_t) value(double) ...
            const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
            
            const std::vector<bool> &vertex_used = tria.get_used_vertices();

            // looping over the nodes of the triangulation
            for(unsigned int i = 0; i < vertex_used.size(); ++i){
                if(!vertex_used[i]) continue;

                dealii::Tensor<1,dim,double> nodeData = data[i];
                
                unsigned int nodeTag = i + 1;
                out << nodeTag;

                // vector entries
                for(unsigned int i = 0; i < dim; ++i)
                    out << " " << nodeData[i];
                
                // padding zeros
                for(unsigned int i = dim; i < 3; ++i)
                    out << " " << 0;

                out << '\n';
            }

            break;

        }case element:{
            // elementTag(size_t) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                dealii::Tensor<1,dim,double> cellData = data[cell->active_cell_index()];
                
                unsigned int elementTag = cell->active_cell_index() + 1;
                out << elementTag;

                // vector entries
                for(unsigned int i = 0; i < dim; ++i)
                    out << " " << cellData[i];

                // padding zeros
                for(unsigned int i = dim; i < 3; ++i)
                    out << " " << 0;

                out << '\n';
            }

            break;

        }case elementNode:{
            // elementTag(size_t) numNodesPerElement(int) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                unsigned int elementTag = cell->active_cell_index() + 1;
                out << elementTag << " " << dealii::GeometryInfo<dim>::vertices_per_cell;

                // looping over the vertices
                for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
                    // switching numbering order to match mesh writing, nodeTag = nodeIndex + 1
                    unsigned int nodeIndex = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]);
                    
                    dealii::Tensor<1,dim,double> nodeData = data[nodeIndex];

                    // vector entries
                    for(unsigned int i = 0; i < dim; ++i)
                        out << " " << nodeData[i];

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
void MshOutDataInternal<PHILIP_DIM, dealii::Tensor<2,PHILIP_DIM,double>>::write_msh_data_internal(
    const dealii::DoFHandler<PHILIP_DIM,PHILIP_DIM> &dof_handler,
    std::ostream &                                   out)
{
    const int dim = PHILIP_DIM;

    // looping through the data structure
    switch(storageType){
        case node:{
            // nodeTag(size_t) value(double) ...
            const dealii::Triangulation<dim,dim> &tria = dof_handler.get_triangulation();
            
            const std::vector<bool> &vertex_used = tria.get_used_vertices();

            // looping over the nodes of the triangulation
            for(unsigned int i = 0; i < vertex_used.size(); ++i){
                if(!vertex_used[i]) continue;

                dealii::Tensor<2,dim,double> nodeData = data[i];
                
                unsigned int nodeTag = i + 1;
                out << nodeTag;

                for(unsigned int i = 0; i < dim; ++i){
                    // matrix entries
                    for(unsigned int j = 0; j < dim; ++j)
                        out << " " << nodeData[i][j];
                    
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

        }case element:{
            // elementTag(size_t) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                dealii::Tensor<2,dim,double> cellData = data[cell->active_cell_index()];
                
                unsigned int elementTag = cell->active_cell_index() + 1;
                out << elementTag;

                for(unsigned int i = 0; i < dim; ++i){
                    // matrix entries
                    for(unsigned int j = 0; j < dim; ++j)
                        out << " " << cellData[i][j];
                    
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

        }case elementNode:{
            // elementTag(size_t) numNodesPerElement(int) value(double) ...
            for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                unsigned int elementTag = cell->active_cell_index() + 1;
                out << elementTag << " " << dealii::GeometryInfo<dim>::vertices_per_cell;

                // looping over the vertices
                for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
                    // switching numbering order to match mesh writing, nodeTag = nodeIndex + 1
                    unsigned int nodeIndex = cell->vertex_index(dealii::GeometryInfo<dim>::ucd_to_deal[vertex]);
                    
                    dealii::Tensor<2,dim,double> nodeData = data[nodeIndex];

                    for(unsigned int i = 0; i < dim; ++i){
                        // matrix entries
                        for(unsigned int j = 0; j < dim; ++j)
                            out << " " << nodeData[i][j];
                        
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

} // namespace GridRefinement

} //namespace PHiLiP