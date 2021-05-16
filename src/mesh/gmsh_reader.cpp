#include <fstream>
#include <map>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_in.h> // Mostly just for their exceptions
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/dofs/dof_renumbering.h>


#include "high_order_grid.h"
#include "gmsh_reader.hpp"


namespace PHiLiP {

/**
* In 1d, boundary indicators are associated with vertices, but this is not
* currently passed through the SubcellData structure. This function sets
* boundary indicators on vertices after the triangulation has already been
* created.
*
* TODO: Fix this properly via SubcellData
*/
template <int spacedim>
void
assign_1d_boundary_ids( const std::map<unsigned int, dealii::types::boundary_id> &boundary_ids,
                        dealii::Triangulation<1, spacedim> &triangulation)
{
    if (boundary_ids.size() > 0) {
        for (const auto &cell : triangulation.active_cell_iterators()) {
            for (unsigned int f : dealii::GeometryInfo<1>::face_indices()) {
                if (boundary_ids.find(cell->vertex_index(f)) != boundary_ids.end()) {

                    AssertThrow( cell->at_boundary(f),
                      dealii::ExcMessage(
                        "You are trying to prescribe boundary ids on the face "
                        "of a 1d cell (i.e., on a vertex), but this face is not actually at "
                        "the boundary of the mesh. This is not allowed."));

                    cell->face(f)->set_boundary_id(boundary_ids.find(cell->vertex_index(f))->second);
                }
            }
        }
    }
}

template <int dim>
void rotate_indices(std::vector<unsigned int> &numbers, const unsigned int n_indices_per_direction, const char direction)
{
  const unsigned int n = n_indices_per_direction;
  unsigned int       s = n;
  for (unsigned int i = 1; i < dim; ++i)
    s *= n;
  numbers.resize(s);

  unsigned int l = 0;

  if (dim == 1)
    {
      // Mirror around midpoint
      for (unsigned int i = n; i > 0;)
        numbers[l++] = --i;
    }
  else
    {
      switch (direction)
        {
          // Rotate xy-plane
          // counter-clockwise
          case 'z':
            for (unsigned int iz = 0; iz < ((dim > 2) ? n : 1); ++iz)
              for (unsigned int j = 0; j < n; ++j)
                for (unsigned int i = 0; i < n; ++i)
                  {
                    unsigned int k = n * i - j + n - 1 + n * n * iz;
                    numbers[l++]   = k;
                  }
            break;
          // Rotate xy-plane
          // clockwise
          case 'Z':
            for (unsigned int iz = 0; iz < ((dim > 2) ? n : 1); ++iz)
              for (unsigned int iy = 0; iy < n; ++iy)
                for (unsigned int ix = 0; ix < n; ++ix)
                  {
                    unsigned int k = n * ix - iy + n - 1 + n * n * iz;
                    numbers[k]     = l++;
                  }
            break;
          // Rotate yz-plane
          // counter-clockwise
          case 'x':
            Assert(dim > 2, dealii::ExcDimensionMismatch(dim, 3));
            for (unsigned int iz = 0; iz < n; ++iz)
              for (unsigned int iy = 0; iy < n; ++iy)
                for (unsigned int ix = 0; ix < n; ++ix)
                  {
                    unsigned int k = n * (n * iy - iz + n - 1) + ix;
                    numbers[l++]   = k;
                  }
            break;
          // Rotate yz-plane
          // clockwise
          case 'X':
            Assert(dim > 2, dealii::ExcDimensionMismatch(dim, 3));
            for (unsigned int iz = 0; iz < n; ++iz)
              for (unsigned int iy = 0; iy < n; ++iy)
                for (unsigned int ix = 0; ix < n; ++ix)
                  {
                    unsigned int k = n * (n * iy - iz + n - 1) + ix;
                    numbers[k]     = l++;
                  }
            break;
          default:
            Assert(false, dealii::ExcNotImplemented());
        }
    }
}

template <int dim, int spacedim>
void
assign_1d_boundary_ids(const std::map<unsigned int, dealii::types::boundary_id> &,
                       dealii::Triangulation<dim, spacedim> &)
{
    // we shouldn't get here since boundary ids are not assigned to
    // vertices except in 1d
    Assert(dim != 1, dealii::ExcInternalError());
}


void open_file_toRead(const std::string filepath, std::ifstream& file_in)
{
    file_in.open(filepath);
    if(!file_in) {
        std::cout << "Could not open file "<< filepath << std::endl;
        std::abort();
    }
}


void read_gmsh_entities(std::ifstream &infile, std::array<std::map<int, int>, 4> &tag_maps)
{
    std::string  line;
    // if the next block is of kind $Entities, parse it
    unsigned long n_points, n_curves, n_surfaces, n_volumes;

    infile >> n_points >> n_curves >> n_surfaces >> n_volumes;
    int entity_tag;
    unsigned int n_physicals;
    double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;
    (void) box_min_x; (void) box_min_y; (void) box_min_z;
    (void) box_max_x; (void) box_max_y; (void) box_max_z;
    for (unsigned int i = 0; i < n_points; ++i) {
        // parse point ids

        // we only care for 'tag' as key for tag_maps[0]
        infile >> entity_tag >> box_min_x >> box_min_y >> box_min_z >> n_physicals;

        // if there is a physical tag, we will use it as boundary id below
        AssertThrow(n_physicals < 2, dealii::ExcMessage("More than one tag is not supported!"));
        // if there is no physical tag, use 0 as default
        int physical_tag = 0;
        for (unsigned int j = 0; j < n_physicals; ++j) {
            infile >> physical_tag;
        }
        //std::cout << "Entity point tag " << entity_tag << " with physical tag " << physical_tag << std::endl;
        tag_maps[0][entity_tag] = physical_tag;
    }
    for (unsigned int i = 0; i < n_curves; ++i) {
        // parse curve ids

        // we only care for 'tag' as key for tag_maps[1]
        infile >> entity_tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
        // if there is a physical tag, we will use it as boundary id below
        AssertThrow(n_physicals < 2, dealii::ExcMessage("More than one tag is not supported!"));
        // if there is no physical tag, use 0 as default
        int physical_tag = 0;
        for (unsigned int j = 0; j < n_physicals; ++j) {
            infile >> physical_tag;
        }
        tag_maps[1][entity_tag] = physical_tag;
        //std::cout << "Entity curve tag " << entity_tag << " with physical tag " << physical_tag << std::endl;
        // we don't care about the points associated to a curve, but have
        // to parse them anyway because their format is unstructured
        infile >> n_points;
        for (unsigned int j = 0; j < n_points; ++j) {
            infile >> entity_tag;
        }
    }
    for (unsigned int i = 0; i < n_surfaces; ++i) {
        // parse surface ids

        // we only care for 'tag' as key for tag_maps[2]
        infile >> entity_tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
        // if there is a physical tag, we will use it as boundary id below
        AssertThrow(n_physicals < 2, dealii::ExcMessage("More than one tag is not supported!"));
        // if there is no physical tag, use 0 as default
        int physical_tag = 0;
        for (unsigned int j = 0; j < n_physicals; ++j) {
          infile >> physical_tag;
        }
        //std::cout << "Entity surface tag " << entity_tag << " with physical tag " << physical_tag << std::endl;
        tag_maps[2][entity_tag] = physical_tag;
        // we don't care about the curves associated to a surface, but
        // have to parse them anyway because their format is unstructured
        infile >> n_curves;
        for (unsigned int j = 0; j < n_curves; ++j) {
          infile >> entity_tag;
        }
    }
    for (unsigned int i = 0; i < n_volumes; ++i) {
        // parse volume ids

        // we only care for 'tag' as key for tag_maps[3]
        infile >> entity_tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
        // if there is a physical tag, we will use it as boundary id below
        AssertThrow(n_physicals < 2,
                    dealii::ExcMessage("More than one tag is not supported!"));
        // if there is no physical tag, use 0 as default
        int physical_tag = 0;
        for (unsigned int j = 0; j < n_physicals; ++j) {
            infile >> physical_tag;
        }
        //std::cout << "Entity volume tag " << entity_tag << " with physical tag " << physical_tag << std::endl;
        tag_maps[3][entity_tag] = physical_tag;
        // we don't care about the surfaces associated to a volume, but
        // have to parse them anyway because their format is unstructured
        infile >> n_surfaces;
        for (unsigned int j = 0; j < n_surfaces; ++j) {
            infile >> entity_tag;
        }
    }
    infile >> line;
    //AssertThrow(line == "$EndEntities", PHiLiP::ExcInvalidGMSHInput(line));
}

template<int spacedim>
void read_gmsh_nodes( std::ifstream &infile, std::vector<dealii::Point<spacedim>> &vertices, std::map<int, int> &vertex_indices )
{
    std::string  line;
    // now read the nodes list
    unsigned int n_entity_blocks, n_vertices;
    int min_node_tag;
    int max_node_tag;
    infile >> n_entity_blocks >> n_vertices >> min_node_tag >> max_node_tag;
    std::cout << "Reading nodes..." << std::endl;
    //std::cout << "Number of entity blocks: " << n_entity_blocks << " with a total of " << n_vertices << " vertices." << std::endl;

    vertices.resize(n_vertices);
  
    unsigned int global_vertex = 0;
    for (unsigned int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
        int           parametric;
        unsigned long numNodes;

        // for gmsh_file_format 4.1 the order of tag and dim is reversed,
        // but we are ignoring both anyway.
        int tagEntity, dimEntity;
        infile >> tagEntity >> dimEntity >> parametric >> numNodes;

        //std::cout << "Entity block: " << entity_block << " with tag " << tagEntity << " in " << dimEntity << " dimension with " << numNodes << " nodes. Parametric: " << parametric << std::endl;

        std::vector<int> vertex_numbers;

        for (unsigned long vertex_per_entity = 0; vertex_per_entity < numNodes; ++vertex_per_entity) {
            int vertex_number;
            infile >> vertex_number;
            //std::cout << vertex_number << std::endl;
            vertex_numbers.push_back(vertex_number);
        }

        for (unsigned long vertex_per_entity = 0; vertex_per_entity < numNodes; ++vertex_per_entity, ++global_vertex) {
            // read vertex
            double x[3];
            infile >> x[0] >> x[1] >> x[2];
            //std::cout << x[0] << " " << x[1] << " " << x[2] << " ";
            //if (parametric == 0) std::cout << std::endl;


            for (unsigned int d = 0; d < spacedim; ++d) {
                vertices[global_vertex](d) = x[d];
            }

            int vertex_number;
            vertex_number = vertex_numbers[vertex_per_entity];
            // store mapping
            vertex_indices[vertex_number] = global_vertex;


            // ignore parametric coordinates
            if (parametric != 0) {
                int n_parametric = dimEntity;
                if (dimEntity == 0) n_parametric = 1;
                double uvw[3];
                for (int d=0; d<n_parametric; ++d) {
                    infile >> uvw[d];
                    //std::cout << uvw[d] << " ";

                }
                //std::cout << std::endl;
                (void)uvw;
            }
        }
    }
    AssertDimension(global_vertex, n_vertices);
    std::cout << "Finished reading nodes." << std::endl;
}

unsigned int gmsh_cell_type_to_order(unsigned int cell_type)
{
    unsigned int cell_order = 0;
    if        ( (cell_type == MSH_LIN_2) || (cell_type == MSH_QUA_4) || (cell_type == MSH_HEX_8) ) {
        cell_order = 1;
    } else if ( (cell_type == MSH_LIN_3) || (cell_type == MSH_QUA_9) || (cell_type == MSH_HEX_27) ) {
        cell_order = 2;
    } else if ( (cell_type == MSH_LIN_4) || (cell_type == MSH_QUA_16) || (cell_type == MSH_HEX_64) ) {
        cell_order = 3;
    } else if ( (cell_type == MSH_LIN_5) || (cell_type == MSH_QUA_25) || (cell_type == MSH_HEX_125) ) {
        cell_order = 4;
    } else if ( (cell_type == MSH_LIN_6) || (cell_type == MSH_QUA_36) || (cell_type == MSH_HEX_216) ) {
        cell_order = 5;
    } else if ( (cell_type == MSH_LIN_7) || (cell_type == MSH_QUA_49) || (cell_type == MSH_HEX_343) ) {
        cell_order = 6;
    } else if ( (cell_type == MSH_LIN_8) || (cell_type == MSH_QUA_64) || (cell_type == MSH_HEX_512) ) {
        cell_order = 7;
    } else if ( (cell_type == MSH_LIN_9) || (cell_type == MSH_QUA_81) || (cell_type == MSH_HEX_729) ) {
        cell_order = 8;
    } else if ( (cell_type == MSH_PNT) ) {
        cell_order = 0;
    } else {
        //AssertThrow(false, dealii::ExcGmshUnsupportedGeometry(cell_type));
    }

    return cell_order;
}

template<int dim>
unsigned int find_grid_order(std::ifstream &infile)
{
    auto entity_file_position = infile.tellg();

    unsigned int grid_order = 0;

    unsigned int n_entity_blocks, n_cells;
    int min_ele_tag, max_ele_tag;
    infile >> n_entity_blocks >> n_cells >> min_ele_tag >> max_ele_tag;

    std::cout << "Finding grid order..." << std::endl;
    std::cout << n_entity_blocks << " entity blocks with a total of " << n_cells << " cells. " << std::endl;

    std::vector<unsigned int> vertices_id;

    unsigned int global_cell = 0;
    for (unsigned int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
        unsigned long numElements;
        int           cell_type;

        int tagEntity, dimEntity;
        infile >> dimEntity >> tagEntity >> cell_type >> numElements;
        //std::cout << "Entity block " << entity_block << " of dimension " << dimEntity << " with tag " << tagEntity << " and celltype = " << cell_type << " containing " << numElements << " elements. " << std::endl;

        const unsigned int cell_order = gmsh_cell_type_to_order(cell_type);
        const unsigned int nodes_per_element = std::pow(cell_order + 1, dimEntity);

        grid_order = std::max(cell_order, grid_order);

        vertices_id.resize(nodes_per_element);

        for (unsigned int cell_per_entity = 0; cell_per_entity < numElements; ++cell_per_entity, ++global_cell) {
            // note that since infile the input file we found the number of p1_cells at the top, there
            // should still be input here, so check this:
            AssertThrow(infile, dealii::ExcIO());

            int tag;
            infile >> tag;

            for (unsigned int i = 0; i < nodes_per_element; ++i) {
                infile >> vertices_id[i];
            }
        } // End of cell per entity
    } // End of entity block

    infile.seekg(entity_file_position);

    AssertDimension(global_cell, n_cells);
    std::cout << "Found grid order = " << grid_order << std::endl;
    return grid_order;
}

unsigned int ijk_to_num(const unsigned int i,
                        const unsigned int j,
                        const unsigned int k,
                        const unsigned int n_per_line)
{
    return i + j*n_per_line + k*n_per_line*n_per_line;
}
unsigned int ij_to_num(const unsigned int i,
                       const unsigned int j,
                       const unsigned int n_per_line)
{
    return i + j*n_per_line;
}

template <int dim>
std::vector<unsigned int>
gmsh_hierarchic_to_lexicographic(const unsigned int degree)
{
    // number of support points in each direction
    const unsigned int n = degree + 1;

    const unsigned int dofs_per_cell = dealii::Utilities::fixed_power<dim>(n);

    std::vector<unsigned int> h2l(dofs_per_cell);
    if (degree == 0) {
        h2l[0] = 0;
        return h2l;
    }

    // polynomial degree
    const unsigned int dofs_per_line = degree - 1;

    // the following lines of code are somewhat odd, due to the way the
    // hierarchic numbering is organized. if someone would really want to
    // understand these lines, you better draw some pictures where you
    // indicate the indices and orders of vertices, lines, etc, along with the
    // numbers of the degrees of freedom in hierarchical and lexicographical
    // order
    switch (dim) {
    case 0: {
        h2l[0] = 0;
        break;
    } case 1: {
        h2l[0] = 0;
        h2l[1] = dofs_per_cell - 1;
        for (unsigned int i = 2; i < dofs_per_cell; ++i)
          h2l[i] = i - 1;
        break;
    } case 2: {
        unsigned int next_index = 0;

        int subdegree = degree;
        int square_reduction = 0;
        while (subdegree > 0) {

            unsigned int start = 0 + square_reduction;
            unsigned int end = n - square_reduction;


            // First the four vertices
            {
                unsigned int i, j;
                // Bottom left
                i = start; j = start;
                h2l[next_index++] = ij_to_num(i,j,n);
                // Bottom right
                i = end-1; j = start;
                h2l[next_index++] = ij_to_num(i,j,n);
                // Top right
                i = end-1; j = end-1;
                h2l[next_index++] = ij_to_num(i,j,n);
                // Top left
                i = start; j = end-1;
                h2l[next_index++] = ij_to_num(i,j,n);
            }

            // Bottom line
            {
                unsigned int j = start;
                for (unsigned int i = start+1; i < end-1; ++i)
                  h2l[next_index++] = ij_to_num(i,j,n);
            }
            // Right line
            {
                unsigned int i = end-1;
                for (unsigned int j = start+1; j < end-1; ++j)
                  h2l[next_index++] = ij_to_num(i,j,n);
            }
            // Top line (right to left)
            {
                unsigned int j = end-1;
                //for (unsigned int i = start+1; i < end-1; ++i)
                // Need signed int otherwise, j=0 followed by --j results in j=UINT_MAX
                for (int i = end-2; i > (int)start; --i)
                  h2l[next_index++] = ij_to_num(i,j,n);
            }
            // Left line (top to bottom order)
            {
                unsigned int i = start;
                //for (unsigned int j = start+1; j < end-1; ++j)
                // Need signed int otherwise, j=0 followed by --j results in j=UINT_MAX
                for (int j = end-2; j > (int)start; --j)
                  h2l[next_index++] = ij_to_num(i,j,n);
            }

            subdegree -= 2;
            square_reduction += 1;

        }
        if (subdegree == 0) {
            const unsigned int middle = (n-1)/2;
            h2l[next_index++] = ij_to_num(middle, middle, n);
        }

        Assert(next_index == dofs_per_cell, dealii::ExcInternalError());

          break;
    } case 3: {

        Assert(false, dealii::ExcNotImplemented());

        unsigned int next_index = 0;
        // first the eight vertices
        h2l[next_index++] = 0;                        // 0
        h2l[next_index++] = (1) * degree;             // 1
        h2l[next_index++] = (n)*degree;               // 2
        h2l[next_index++] = (n + 1) * degree;         // 3
        h2l[next_index++] = (n * n) * degree;         // 4
        h2l[next_index++] = (n * n + 1) * degree;     // 5
        h2l[next_index++] = (n * n + n) * degree;     // 6
        h2l[next_index++] = (n * n + n + 1) * degree; // 7

        // line 0
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = (i + 1) * n;
        // line 1
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = n - 1 + (i + 1) * n;
        // line 2
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = 1 + i;
        // line 3
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = 1 + i + n * (n - 1);

        // line 4
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = (n - 1) * n * n + (i + 1) * n;
        // line 5
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = (n - 1) * (n * n + 1) + (i + 1) * n;
        // line 6
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = n * n * (n - 1) + i + 1;
        // line 7
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = n * n * (n - 1) + i + 1 + n * (n - 1);

        // line 8
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = (i + 1) * n * n;
        // line 9
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = n - 1 + (i + 1) * n * n;
        // line 10
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = (i + 1) * n * n + n * (n - 1);
        // line 11
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          h2l[next_index++] = n - 1 + (i + 1) * n * n + n * (n - 1);


        // inside quads
        // face 0
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            h2l[next_index++] = (i + 1) * n * n + n * (j + 1);
        // face 1
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            h2l[next_index++] = (i + 1) * n * n + n - 1 + n * (j + 1);
        // face 2, note the orientation!
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            h2l[next_index++] = (j + 1) * n * n + i + 1;
        // face 3, note the orientation!
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            h2l[next_index++] = (j + 1) * n * n + n * (n - 1) + i + 1;
        // face 4
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            h2l[next_index++] = n * (i + 1) + j + 1;
        // face 5
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            h2l[next_index++] = (n - 1) * n * n + n * (i + 1) + j + 1;

        // inside hex
        for (unsigned int i = 0; i < dofs_per_line; ++i)
          for (unsigned int j = 0; j < dofs_per_line; ++j)
            for (unsigned int k = 0; k < dofs_per_line; ++k)
              h2l[next_index++] = n * n * (i + 1) + n * (j + 1) + k + 1;

        Assert(next_index == dofs_per_cell, dealii::ExcInternalError());

        break;
    } default: {
        Assert(false, dealii::ExcNotImplemented());
    }
    
    } // End of switch

    return h2l;
}

unsigned int dealii_node_number(const unsigned int i,
                                const unsigned int j,
                                const unsigned int k)
{
    return i+j+k;
}

void fe_q_node_number(const unsigned int index,
                                unsigned int &i,
                                unsigned int &j,
                                unsigned int &k)
{
    i = index;
    j = index;
    k = index;
}

template <int dim, int spacedim>
std::shared_ptr< HighOrderGrid<dim, double> >
read_gmsh(std::string filename, int requested_grid_order)
{

    //for (unsigned int deg = 1; deg < 7; ++deg) {
    //    std::cout << "DEGREE " << deg << std::endl;
    //    std::vector<unsigned int> h2l = gmsh_hierarchic_to_lexicographic<dim>(deg);
    //    std::vector<unsigned int> l2h = dealii::Utilities::invert_permutation(h2l);

    //    unsigned int n = deg+1;
    //    std::cout << "L2H "  << std::endl;
    //    for (int j=n-1; j>=0; --j) {
    //        for (unsigned int i=0; i<n; ++i) {
    //            const unsigned int ij = ij_to_num(i,j,n);
    //            std::cout << l2h[ij] << " ";
    //        }
    //        std::cout << std::endl;
    //    }

    //    std::cout << std::endl << std::endl;
    //}
    //std::abort();

    Assert(dim==2, dealii::ExcInternalError());
    std::ifstream infile;

    open_file_toRead(filename, infile);
  
    std::string  line;
    // This array stores maps from the 'entities' to the 'physical tags' for
    // points, curves, surfaces and volumes. We use this information later to
    // assign boundary ids.
    std::array<std::map<int, int>, 4> tag_maps;
  
  
    infile >> line;
  
    //Assert(tria != nullptr, dealii::ExcNoTriangulationSelected());
  
    // first determine file format
    unsigned int gmsh_file_format = 0;
    if (line == "$MeshFormat") {
      gmsh_file_format = 20;
    } else {
      //AssertThrow(false, dealii::ExcInvalidGMSHInput(line));
    }
  
  
    // if file format is 2.0 or greater then we also have to read the rest of the
    // header
    if (gmsh_file_format == 20) {
        double       version;
        unsigned int file_type, data_size;
  
        infile >> version >> file_type >> data_size;
  
        Assert((version == 4.1), dealii::ExcNotImplemented());
        gmsh_file_format = static_cast<unsigned int>(version * 10);
  
  
        Assert(file_type == 0, dealii::ExcNotImplemented());
        Assert(data_size == sizeof(double), dealii::ExcNotImplemented());
  
  
        // read the end of the header and the first line of the nodes description
        // to synch ourselves with the format 1 handling above
        infile >> line;
        //AssertThrow(line == "$EndMeshFormat", PHiLiP::ExcInvalidGMSHInput(line));
  
  
        infile >> line;
        // if the next block is of kind $PhysicalNames, ignore it
        if (line == "$PhysicalNames") {
            do {
                infile >> line;
            } while (line != "$EndPhysicalNames");
            infile >> line;
        }
  
  
        // if the next block is of kind $Entities, parse it
        if (line == "$Entities") read_gmsh_entities(infile, tag_maps);
        infile >> line;
  
        // if the next block is of kind $PartitionedEntities, ignore it
        if (line == "$PartitionedEntities") {
            do {
                infile >> line;
            } while (line != "$EndPartitionedEntities");
            infile >> line;
        }
  
        // but the next thing should,
        // infile any case, be the list of
        // nodes:
        //AssertThrow(line == "$Nodes", PHiLiP::ExcInvalidGMSHInput(line));
    }
  
    std::vector<dealii::Point<spacedim>> vertices;
    // set up mapping between numbering
    // infile msh-file (nod) and infile the
    // vertices vector
    std::map<int, int> vertex_indices;
    read_gmsh_nodes( infile, vertices, vertex_indices );
  
    // Assert we reached the end of the block
    infile >> line;
    static const std::string end_nodes_marker = "$EndNodes";
    //AssertThrow(line == end_nodes_marker, PHiLiP::ExcInvalidGMSHInput(line));
  
    // Now read infile next bit
    infile >> line;
    static const std::string begin_elements_marker = "$Elements";
    //AssertThrow(line == begin_elements_marker, PHiLiP::ExcInvalidGMSHInput(line));

    const unsigned int grid_order = find_grid_order<dim>(infile);
  
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> triangulation = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    auto high_order_grid = std::make_shared<HighOrderGrid<dim, double>>(grid_order, triangulation);
  
    unsigned int n_entity_blocks, n_cells;
    int min_ele_tag, max_ele_tag;
    infile >> n_entity_blocks >> n_cells >> min_ele_tag >> max_ele_tag;
    // set up array of p1_cells and subcells (faces). In 1d, there is currently no
    // standard way infile deal.II to pass boundary indicators attached to individual
    // vertices, so do this by hand via the boundary_ids_1d array

    std::vector<dealii::CellData<dim>> p1_cells;
    std::vector<dealii::CellData<dim>> high_order_cells;
    dealii::CellData<dim> temp_high_order_cells;

    dealii::SubCellData                                subcelldata;
    std::map<unsigned int, dealii::types::boundary_id> boundary_ids_1d;

  
  
    unsigned int global_cell = 0;
    for (unsigned int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
        unsigned int  material_id;
        unsigned long numElements;
        int           cell_type;

        // for gmsh_file_format 4.1 the order of tag and dim is reversed,
        int tagEntity, dimEntity;
        infile >> dimEntity >> tagEntity >> cell_type >> numElements;
        material_id = tag_maps[dimEntity][tagEntity];

        const unsigned int cell_order = gmsh_cell_type_to_order(cell_type);

        unsigned int vertices_per_element = std::pow(2, dimEntity);
        unsigned int nodes_per_element = std::pow(cell_order + 1, dimEntity);


        for (unsigned int cell_per_entity = 0; cell_per_entity < numElements; ++cell_per_entity, ++global_cell) {

            // ignore tag
            int tag;
            infile >> tag;

            if (dimEntity == dim) {
                // Found a cell

                // Allocate and read indices
                p1_cells.emplace_back(vertices_per_element);
                high_order_cells.emplace_back(vertices_per_element);

                auto &p1_vertices_id = p1_cells.back().vertices;
                auto &high_order_vertices_id = high_order_cells.back().vertices;

                p1_vertices_id.resize(vertices_per_element);
                high_order_vertices_id.resize(nodes_per_element);

                for (unsigned int i = 0; i < nodes_per_element; ++i) {
                    infile >> high_order_vertices_id[i];
                }
                for (unsigned int i = 0; i < vertices_per_element; ++i) {
                    p1_vertices_id[i] = high_order_vertices_id[i];
                }

                // to make sure that the cast won't fail
                Assert(material_id <= std::numeric_limits<dealii::types::material_id>::max(),
                       dealii::ExcIndexRange( material_id, 0, std::numeric_limits<dealii::types::material_id>::max()));
                // we use only material_ids infile the range from 0 to dealii::numbers::invalid_material_id-1
                AssertIndexRange(material_id, dealii::numbers::invalid_material_id);

                p1_cells.back().material_id = material_id;

                // transform from ucd to consecutive numbering
                for (unsigned int i = 0; i < vertices_per_element; ++i) {
                    //AssertThrow( vertex_indices.find(p1_cells.back().vertices[i]) != vertex_indices.end(),
                    //  dealii::ExcInvalidVertexIndexGmsh(global_cell, elm_number, p1_cells.back().vertices[i]));

                    // vertex with this index exists
                    p1_vertices_id[i] = vertex_indices[p1_cells.back().vertices[i]];
                }
                for (unsigned int i = 0; i < nodes_per_element; ++i) {
                    high_order_vertices_id[i] = vertex_indices[high_order_cells.back().vertices[i]];
                }
            } else if (dimEntity == 1 && dimEntity < dim) {
                // Boundary info
                subcelldata.boundary_lines.emplace_back(vertices_per_element);
                auto &p1_vertices_id = subcelldata.boundary_lines.back().vertices;
                p1_vertices_id.resize(vertices_per_element);

                temp_high_order_cells.vertices.resize(nodes_per_element);

                for (unsigned int i = 0; i < nodes_per_element; ++i) {
                    infile >> temp_high_order_cells.vertices[i];
                }
                for (unsigned int i = 0; i < vertices_per_element; ++i) {
                    p1_vertices_id[i] = temp_high_order_cells.vertices[i];
                }

                // to make sure that the cast won't fail
                Assert(material_id <= std::numeric_limits<dealii::types::boundary_id>::max(),
                       dealii::ExcIndexRange( material_id, 0, std::numeric_limits<dealii::types::boundary_id>::max()));
                // we use only boundary_ids infile the range from 0 to dealii::numbers::internal_face_boundary_id-1
                AssertIndexRange(material_id, dealii::numbers::internal_face_boundary_id);

                subcelldata.boundary_lines.back().boundary_id = static_cast<dealii::types::boundary_id>(material_id);

                // transform from ucd to consecutive numbering
                for (unsigned int &vertex : subcelldata.boundary_lines.back().vertices) {
                    if (vertex_indices.find(vertex) != vertex_indices.end()) {
                      vertex = vertex_indices[vertex];
                    } else {
                        // no such vertex index
                        //AssertThrow(false, dealii::ExcInvalidVertexIndex(cell_per_entity, vertex));
                        vertex = dealii::numbers::invalid_unsigned_int;
                        std::abort();
                    }
                }
            } else if (dimEntity == 2 && dimEntity < dim) {
                // boundary info
                subcelldata.boundary_quads.emplace_back(vertices_per_element);
                auto &p1_vertices_id = subcelldata.boundary_quads.back().vertices;
                p1_vertices_id.resize(vertices_per_element);

                temp_high_order_cells.vertices.resize(nodes_per_element);

                for (unsigned int i = 0; i < nodes_per_element; ++i) {
                    infile >> temp_high_order_cells.vertices[i];
                }
                for (unsigned int i = 0; i < vertices_per_element; ++i) {
                    p1_vertices_id[i] = temp_high_order_cells.vertices[i];
                }

                // to make sure that the cast won't fail
                Assert(material_id <= std::numeric_limits<dealii::types::boundary_id>::max(),
                       dealii::ExcIndexRange( material_id, 0, std::numeric_limits<dealii::types::boundary_id>::max()));
                // we use only boundary_ids infile the range from 0 to dealii::numbers::internal_face_boundary_id-1
                AssertIndexRange(material_id, dealii::numbers::internal_face_boundary_id);

                subcelldata.boundary_quads.back().boundary_id = static_cast<dealii::types::boundary_id>(material_id);

                // transform from gmsh to consecutive numbering
                for (unsigned int &vertex : subcelldata.boundary_quads.back().vertices) {
                    if (vertex_indices.find(vertex) != vertex_indices.end()) {
                      vertex = vertex_indices[vertex];
                    } else {
                        // no such vertex index
                        //Assert(false, dealii::ExcInvalidVertexIndex(cell_per_entity, vertex));
                        vertex = dealii::numbers::invalid_unsigned_int;
                    }
                }
            } else if (cell_type == MSH_PNT) {
              // read the indices of nodes given
              unsigned int node_index = 0;
              infile >> node_index;

              // We only care about boundary indicators assigned to individual
              // vertices infile 1d (because otherwise the vertices are not faces)
              if (dim == 1) {
                  boundary_ids_1d[vertex_indices[node_index]] = material_id;
              }
            } else {
              //AssertThrow(false, dealii::ExcGmshUnsupportedGeometry(cell_type));
            }
        } // End of cell per entity
    } // End of entity block
    AssertDimension(global_cell, n_cells);

    // Assert we reached the end of the block
    infile >> line;
    static const std::string end_elements_marker[] = {"$ENDELM", "$EndElements"};
    //AssertThrow(line == end_elements_marker[gmsh_file_format == 10 ? 0 : 1],
    //            PHiLiP::ExcInvalidGMSHInput(line));
  
    // check that no forbidden arrays are used
    Assert(subcelldata.check_consistency(dim), dealii::ExcInternalError());
  
  
    AssertThrow(infile, dealii::ExcIO());
  
  
    // // check that we actually read some p1_cells.
    // AssertThrow(p1_cells.size() > 0, dealii::ExcGmshNoCellInformation());
  
    // do some clean-up on vertices...
    auto all_vertices = vertices;
    dealii::GridTools::delete_unused_vertices(vertices, p1_cells, subcelldata);
    // ... and p1_cells
    if (dim == spacedim) {
      dealii::GridReordering<dim, spacedim>::invert_all_cells_of_negative_grid(vertices, p1_cells);
    }
    dealii::GridReordering<dim, spacedim>::reorder_cells(p1_cells);
    triangulation->create_triangulation_compatibility(vertices, p1_cells, subcelldata);

    triangulation->repartition();

    dealii::GridOut gridout;
    gridout.write_mesh_per_processor_as_vtu(*(high_order_grid->triangulation), "tria");
  
    // in 1d, we also have to attach boundary ids to vertices, which does not
    // currently work through the call above
    if (dim == 1) {
        assign_1d_boundary_ids(boundary_ids_1d, *triangulation);
    }

    high_order_grid->initialize_with_triangulation_manifold();

    std::vector<unsigned int> deal_h2l = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_order);
    std::vector<unsigned int> deal_l2h = dealii::Utilities::invert_permutation(deal_h2l);

    std::vector<unsigned int> gmsh_h2l = gmsh_hierarchic_to_lexicographic<dim>(grid_order);
    std::vector<unsigned int> gmsh_l2h = dealii::Utilities::invert_permutation(gmsh_h2l);

    // Visualize indexing
    // for (unsigned int deg = 1; deg < 7; ++deg) {
    //     std::cout << "DEGREE " << deg << std::endl;
    //     {
    //         std::vector<unsigned int> h2l = gmsh_hierarchic_to_lexicographic<dim>(deg);
    //         std::vector<unsigned int> l2h = dealii::Utilities::invert_permutation(h2l);

    //         unsigned int n = deg+1;
    //         std::cout << "GMSH L2H "  << std::endl;
    //         for (int j=n-1; j>=0; --j) {
    //             for (unsigned int i=0; i<n; ++i) {
    //                 const unsigned int ij = ij_to_num(i,j,n);
    //                 std::cout << l2h[ij] << " ";
    //             }
    //             std::cout << std::endl;
    //         }

    //         std::cout << std::endl << std::endl;
    //     }
    //     {
    //         std::vector<unsigned int> h2l = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(deg);
    //         std::vector<unsigned int> l2h = dealii::Utilities::invert_permutation(h2l);

    //         unsigned int n = deg+1;
    //         std::cout << "DEAL L2H "  << std::endl;
    //         for (int j=n-1; j>=0; --j) {
    //             for (unsigned int i=0; i<n; ++i) {
    //                 const unsigned int ij = ij_to_num(i,j,n);
    //                 std::cout << l2h[ij] << " ";
    //             }
    //             std::cout << std::endl;
    //         }

    //         std::cout << std::endl << std::endl;
    //     }
    // }
    // std::abort();

    int icell = 0;
    std::vector<dealii::types::global_dof_index> dof_indices(high_order_grid->fe_system.dofs_per_cell);

    //for (unsigned int i=0; i<all_vertices.size(); ++i) {
    //    std::cout << " i " << i 
    //              << " maps to global id " 
    //              << " vertices high_order_vertices_id[i] << " point " << all_vertices[high_order_vertices_id[i]] << std::endl;
    //}
    for (const auto &cell : high_order_grid->dof_handler_grid.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            auto &high_order_vertices_id = high_order_cells[icell].vertices;

            //for (unsigned int i=0; i<high_order_vertices_id.size(); ++i) {
            //    std::cout << " I " << high_order_vertices_id[i] << " point " << all_vertices[high_order_vertices_id[i]] << std::endl;
            //}
            cell->get_dof_indices(dof_indices);

            //std::cout << " cell " << icell << " with vertices: " << std::endl;
            //for (unsigned int i=0; i < cell->n_vertices(); ++i) {
            //    std::cout << cell->vertex(i) << std::endl;
            //}
            //std::cout << " highordercell with vertices: " << std::endl;

            std::vector<unsigned int> rotate_z90degree;
            rotate_indices<dim>(rotate_z90degree, grid_order+1, 'Z');

            bool good_rotation = false;

            auto high_order_vertices_id_lexico = high_order_vertices_id;
            for (unsigned int ihierachic=0; ihierachic<high_order_vertices_id.size(); ++ihierachic) {
                const unsigned int lexico_id = gmsh_h2l[ihierachic];
                high_order_vertices_id_lexico[lexico_id] = high_order_vertices_id[ihierachic];
            }

            //auto high_order_vertices_id_rotated = high_order_cells[icell].vertices;
            auto high_order_vertices_id_rotated = high_order_vertices_id_lexico;
            for (int zr = 0; zr < 4; ++zr) {
                
                std::vector<int> matching(cell->n_vertices());
                for (unsigned int i_vertex=0; i_vertex < cell->n_vertices(); ++i_vertex) {

                    const unsigned int base_index = i_vertex;
                    const unsigned int lexicographic_index = deal_h2l[base_index];

                    //const unsigned int gmsh_hierarchical_index = gmsh_l2h[lexicographic_index];
                    //const unsigned int vertex_id = high_order_vertices_id_rotated[gmsh_hierarchical_index];
                    const unsigned int vertex_id = high_order_vertices_id_rotated[lexicographic_index];
                    const dealii::Point<dim,double> high_order_vertex = all_vertices[vertex_id];
                    //std::cout << high_order_vertex << std::endl;

                    bool found = false;
                    for (unsigned int i=0; i < cell->n_vertices(); ++i) {
                        if (cell->vertex(i) == high_order_vertex) {
                            //std::cout << " cell vertex " << i << " matches HO vertex " << i_vertex << std::endl;
                            found = true;
                        }
                    }
                    if (!found) {
                        std::cout << "Wrong cell... High order nodes do not match the cell's vertices." << std::endl;
                        std::abort();
                    }

                    matching[i_vertex] = (high_order_vertex == cell->vertex(i_vertex)) ? 0 : 1;
                }

                bool all_matching = true;
                for (unsigned int i=0; i < cell->n_vertices(); ++i) {
                    if (matching[i] == 1) all_matching = false;
                }
                good_rotation = all_matching;
                if (good_rotation) {
                    //std::cout << "Found rotation cell..." << std::endl;
                    break;
                }

                //std::cout << "Rotating indices " << std::endl;
                high_order_vertices_id = high_order_vertices_id_rotated;
                for (unsigned int i=0; i<high_order_vertices_id.size(); ++i) {
                    high_order_vertices_id_rotated[i] = high_order_vertices_id[rotate_z90degree[i]];
                }
            }
            if (!good_rotation) {
                std::cout << "Couldn't find rotation..." << std::endl;
                std::abort();
            } 



            for (unsigned int i_vertex = 0; i_vertex < high_order_vertices_id.size(); ++i_vertex) {

                const unsigned int base_index = i_vertex;
                const unsigned int lexicographic_index = deal_h2l[base_index];
                //const unsigned int gmsh_hierarchical_index = gmsh_l2h[lexicographic_index];
                //const unsigned int vertex_id = high_order_vertices_id[gmsh_hierarchical_index];
                const unsigned int vertex_id = high_order_vertices_id_rotated[lexicographic_index];
                const dealii::Point<dim,double> vertex = all_vertices[vertex_id];

                //std::cout << "i_vertex " << i_vertex << " point: " << vertex << std::endl;

                for (int d = 0; d < dim; ++d) {
                    const unsigned int comp = d;
                    const unsigned int shape_index = high_order_grid->dof_handler_grid.get_fe().component_to_system_index(comp, base_index);
                    const unsigned int idof_global = dof_indices[shape_index];

                    //std::cout << " icell " << icell
                    //          << " i_vertex " << i_vertex
                    //          << " i_dim " << d
                    //          << " base_index " << base_index
                    //          << " shape_index " << shape_index 
                    //          << " idof_global " << idof_global
                    //          << " lexicographic_index " << lexicographic_index
                    //          //<< " gmsh_hierarchical_index " << gmsh_hierarchical_index
                    //          << " vertex_id " << vertex_id
                    //          << std::endl;
                    high_order_grid->volume_nodes[idof_global] = vertex[d];
                }
                //std::cout << std::endl;
            }
        }
        icell++;
    }
    high_order_grid->volume_nodes.update_ghost_values();
    high_order_grid->ensure_conforming_mesh();

    
    /// Convert the equidistant points from Gmsh into the GLL points used by FE_Q in deal.II.
    {
        std::vector<dealii::Point<1>> equidistant_points(grid_order+1);
        const double dx = 1.0 / grid_order;
        for (unsigned int i=0; i<grid_order+1; ++i) {
            equidistant_points[i](0) = i*dx;
        }
        dealii::Quadrature<1> quad_equidistant(equidistant_points);
        dealii::FE_Q<dim> fe_q_equidistant(quad_equidistant);
        dealii::FESystem<dim> fe_system_equidistant(fe_q_equidistant, dim);
        dealii::DoFHandler<dim> dof_handler_equidistant(*triangulation);

        dof_handler_equidistant.initialize(*triangulation, fe_system_equidistant);
        dof_handler_equidistant.distribute_dofs(fe_system_equidistant);
        dealii::DoFRenumbering::Cuthill_McKee(dof_handler_equidistant);

        auto equidistant_nodes = high_order_grid->volume_nodes;
        equidistant_nodes.update_ghost_values();
        high_order_grid->volume_nodes.update_ghost_values();
        dealii::FETools::interpolate(dof_handler_equidistant, equidistant_nodes, high_order_grid->dof_handler_grid, high_order_grid->volume_nodes);
        high_order_grid->volume_nodes.update_ghost_values();
        high_order_grid->ensure_conforming_mesh();
    }

    high_order_grid->update_surface_nodes();
    high_order_grid->update_mapping_fe_field();
    high_order_grid->output_results_vtk(9999);
    high_order_grid->reset_initial_nodes();
    
    //return high_order_grid;

    if (requested_grid_order > 0) {
        auto grid = std::make_shared<HighOrderGrid<dim, double>>(requested_grid_order, triangulation);
        grid->initialize_with_triangulation_manifold();
        
        /// Convert the mesh by interpolating from one order to another.
        {
            std::vector<dealii::Point<1>> equidistant_points(grid_order+1);
            const double dx = 1.0 / grid_order;
            for (unsigned int i=0; i<grid_order+1; ++i) {
                equidistant_points[i](0) = i*dx;
            }
            dealii::Quadrature<1> quad_equidistant(equidistant_points);
            dealii::FE_Q<dim> fe_q_equidistant(quad_equidistant);
            dealii::FESystem<dim> fe_system_equidistant(fe_q_equidistant, dim);
            dealii::DoFHandler<dim> dof_handler_equidistant(*triangulation);

            dof_handler_equidistant.initialize(*triangulation, fe_system_equidistant);
            dof_handler_equidistant.distribute_dofs(fe_system_equidistant);
            dealii::DoFRenumbering::Cuthill_McKee(dof_handler_equidistant);

            auto equidistant_nodes = high_order_grid->volume_nodes;
            equidistant_nodes.update_ghost_values();
            grid->volume_nodes.update_ghost_values();
            dealii::FETools::interpolate(dof_handler_equidistant, equidistant_nodes, grid->dof_handler_grid, grid->volume_nodes);
            grid->volume_nodes.update_ghost_values();
            grid->ensure_conforming_mesh();
        }
        grid->update_surface_nodes();
        grid->update_mapping_fe_field();
        grid->output_results_vtk(9999);
        grid->reset_initial_nodes();

        return grid;
    } else {
        return high_order_grid;
    }
}


#if PHILIP_DIM==1 
#else
template std::shared_ptr< HighOrderGrid<PHILIP_DIM, double> > read_gmsh<PHILIP_DIM,PHILIP_DIM>(std::string filename, int requested_grid_order);
#endif

} // namespace PHiLiP
