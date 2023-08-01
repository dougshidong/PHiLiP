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
void rotate_indices(std::vector<unsigned int> &numbers, const unsigned int n_indices_per_direction, const char direction, const bool mesh_reader_verbose_output)
{

    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

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
  else if (dim == 2)                                
  {
      switch (direction)
        {
          // Rotate xy-plane
          // counter-clockwise
          // 3 6 2           2 5 1
          // 7 8 5  becomes  6 8 4
          // 0 4 1           3 7 0
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
          // 3 6 2           0 7 3
          // 7 8 5  becomes  4 8 6
          // 0 4 1           1 5 2
          case 'Z':
            for (unsigned int iz = 0; iz < ((dim > 2) ? n : 1); ++iz)
              for (unsigned int iy = 0; iy < n; ++iy)
                for (unsigned int ix = 0; ix < n; ++ix)
                  {
                    unsigned int k = n * ix - iy + n - 1 + n * n * iz;
                    numbers[k]     = l++;
                  }
            break;
          // Change Z normal
          // Instead of 
          // 3 6 2           1 5 2
          // 7 8 5  becomes  4 8 6
          // 0 4 1           0 7 3
          case '3':
            for (unsigned int iz = 0; iz < ((dim > 2) ? n : 1); ++iz)
              for (unsigned int iy = 0; iy < n; ++iy)
                for (unsigned int ix = 0; ix < n; ++ix)
                  {
                    unsigned int k = iy + n * ix + n * n * iz; // transpose x and y indices
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
    } else {

      //3D ROTATION
      /**
       * Can have 4 possibilities:
       * 1) Rotate about Z-Axis in clockwise direction
       * 2) Rotate about X-Axis in clockwise direction
       * 3) Rotate about Y-Axis in clockwise direction
       * 4) Flip the entire cube, i.e.,
       *      
       *            Rotate Z-Axis
       *               - - - -> 
       *              -       
       *             - - - -
       *                 .
       *                 .
       *                 . 
       *                 7-----------8
       *                -           --
       *               -           - -         ^  .
       *              3----------4   -         .  .
       *              -          -   -         .  .
       *              -          -   6 ....... .  ..... Rotate Y-Axis
       *              -          -  -          . .
       *              -          - -           .  
       *              1----------2
       *              .
       *       .- - - ->     
       *       .    .
       *       .   .
       *       .  .
       *       - - - - -
       *        .
       *       .
       *  Rotate X-Axis
       * 
       * 
       *          --------------------------------
       *          /flip to below, similar to
       *           mirroring it about X-Z plane/
       *          --------------------------------
       * 
       * 
       *            Rotate Z-Axis
       *               - - - -> 
       *              -       
       *             - - - -
       *                 .
       *                 .
       *                 . 
       *                 8-----------7
       *                -           --
       *               -           - -         ^  .
       *              4----------3   -         .  .
       *              -          -   -         .  .
       *              -          -   5 ....... .  ..... Rotate Y-Axis
       *              -          -  -          . .
       *              -          - -           .  
       *              2----------1
       *              .
       *       .- - - ->     
       *       .    .
       *       .   .
       *       .  .
       *       - - - - -
       *        .
       *       .
       *  Rotate X-Axis
       * 
       * 
       * 
       *        ---------------------
       *        /Then, rotate again!/
       *        ---------------------
       * 
       * 
       **/ 

      switch (direction)                                                     
      {
          // Rotate Cube in Z-Axis
          case 'Z':
              for (unsigned int iz = 0; iz < n; ++iz)
                  for (unsigned int iy = 0; iy < n; ++iy)
                      for (unsigned int ix = 0; ix < n; ++ix)
                      {
                          unsigned int k = (ix) * n + n - (iy + 1) + (n * n * iz);
                          numbers[k]     = l++;
                          if(mesh_reader_verbose_output) pcout << "3D rotation matrix, physical node mapping, Z-axis : " << k << std::endl;
                      }
              break;
          // Rotate Cube in X-Axis
          case 'X':
              for (unsigned int iz = 0; iz < n; ++iz)
                  for (unsigned int iy = 0; iy < n; ++iy)
                      for (unsigned int ix = 0; ix < n; ++ix)
                      {
                          unsigned int k = (ix) + (n * (n-1)) + (n * n * iy) - (n * iz);
                          numbers[k] = l++;
                          if(mesh_reader_verbose_output) pcout << "3D rotation matrix, physical node mapping, X-axis : " << k << std::endl;
                      }
              break;
          // Rotate Cube in Y-Axis
          case 'Y':
              for (unsigned int iz = 0; iz < n; ++iz)
                  for (unsigned int iy = 0; iy < n; ++iy)
                      for (unsigned int ix = 0; ix < n; ++ix)
                      {
                          unsigned int k = (ix * n * n) + (n - 1) + (iy * n) - (iz);
                          numbers[k] = l++;
                          if(mesh_reader_verbose_output) pcout << "3D rotation matrix, physical node mapping, Y-axis : " << k << std::endl;
                      }
              break;
          // Flip Cube
          case 'F':
              for (unsigned int iz = 0; iz < n; ++iz)
                  for (unsigned int iy = 0; iy < n; ++iy)
                      for (unsigned int ix = 0; ix < n; ++ix)
                      {
                          unsigned int k = (n * (n - 1)) + ix - (iy * n) + (n * n * iz);
                          numbers[k] = l++;
                          if(mesh_reader_verbose_output) pcout << "3D rotation matrix, physical node mapping, Flip-axis : " << k << std::endl;
                      }
              break;
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
void read_gmsh_nodes( std::ifstream &infile, std::vector<dealii::Point<spacedim>> &vertices, std::map<int, int> &vertex_indices, const bool mesh_reader_verbose_output )
{

    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    std::string  line;
    // now read the nodes list
    unsigned int n_entity_blocks, n_vertices;
    int min_node_tag;
    int max_node_tag;
    infile >> n_entity_blocks >> n_vertices >> min_node_tag >> max_node_tag;
    if(mesh_reader_verbose_output) pcout << "Reading nodes..." << std::endl;

    vertices.resize(n_vertices);
  
    unsigned int global_vertex = 0;
    for (unsigned int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
        int           parametric;
        unsigned long numNodes;

        // for gmsh_file_format 4.1 the order of tag and dim is reversed,
        // but we are ignoring both anyway.
        int tagEntity, dimEntity;
        infile >> tagEntity >> dimEntity >> parametric >> numNodes;

        std::vector<int> vertex_numbers;

        for (unsigned long vertex_per_entity = 0; vertex_per_entity < numNodes; ++vertex_per_entity) {
            int vertex_number;
            infile >> vertex_number;
            vertex_numbers.push_back(vertex_number);
        }

        for (unsigned long vertex_per_entity = 0; vertex_per_entity < numNodes; ++vertex_per_entity, ++global_vertex) {
            // read vertex
            double x[3];
            infile >> x[0] >> x[1] >> x[2];

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
                }
                (void)uvw;
            }
        }
    }
    AssertDimension(global_vertex, n_vertices);
    if(mesh_reader_verbose_output) pcout << "Finished reading nodes." << std::endl;
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
        std::cout << "Invalid element type read from GMSH " << cell_type << ". "
                  << "\n Valid element types are:"
                  << "\n " << MSH_PNT
                  << "\n " << MSH_LIN_2 << " " << MSH_LIN_3 << " " << MSH_LIN_4 << " " << MSH_LIN_5 << " " << MSH_LIN_6 << " " << MSH_LIN_7 << " " << MSH_LIN_8 << " " << MSH_LIN_9
                  << "\n " << MSH_QUA_4 << " " << MSH_QUA_9 << " " << MSH_QUA_16 << " " << MSH_QUA_25 << " " << MSH_QUA_36 << " " << MSH_QUA_49 << " " << MSH_QUA_64 << " " << MSH_QUA_81
                  << "\n " << MSH_HEX_8 <<  " " << MSH_HEX_27 << " " << MSH_HEX_64 << " " << MSH_HEX_125 << " " << MSH_HEX_216 << " " << MSH_HEX_343 << " " << MSH_HEX_512 << " " << MSH_HEX_729
                  << std::endl;
        std::abort();
    }

    return cell_order;
}

/**
 * 
 * Finds the grid order
 * 
 **/ 
template<int dim>
unsigned int find_grid_order(std::ifstream &infile,const bool mesh_reader_verbose_output)
{

    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    auto entity_file_position = infile.tellg();

    unsigned int grid_order = 0;

    unsigned int n_entity_blocks, n_cells;
    int min_ele_tag, max_ele_tag;
    infile >> n_entity_blocks >> n_cells >> min_ele_tag >> max_ele_tag;

    if(mesh_reader_verbose_output) pcout << "Finding grid order..." << std::endl;
    if(mesh_reader_verbose_output) pcout << n_entity_blocks << " entity blocks with a total of " << n_cells << " cells. " << std::endl;

    std::vector<unsigned int> vertices_id;

    unsigned int global_cell = 0;
    for (unsigned int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
        unsigned long numElements;
        int           cell_type;

        int tagEntity, dimEntity;
        infile >> dimEntity >> tagEntity >> cell_type >> numElements;

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
    if(mesh_reader_verbose_output) pcout << "Found grid order = " << grid_order << std::endl;
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

/**
 * Solely used in the 3D case. Helps finding the face_nodes during recursive call.
 * @param degree (Degree of the inner cube)
 *               -> Degree is -2 in this function call because we lose two points
 *                  when we deal with the inner faces/cube.
 *                  -> 1D example: i.e. (1-2-3-4-5), 4th order line with 5 points, 
 *                     if we switch to inner line, we obtain (2-3-4), hence, 
 *                     we obtain a 2nd order line with 3 points.
 * 
 */
std::vector<unsigned int>
face_node_finder(const unsigned int degree, const bool mesh_reader_verbose_output)
{

    /**
    * For Debug output
    */
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    // number of support points in each direction
    const unsigned int n = degree + 1;

    const unsigned int dofs_per_face = dealii::Utilities::fixed_power<2>(n);

    std::vector<unsigned int> h2l_2D(dofs_per_face);
    if (degree == 0) {
        h2l_2D[0] = 0;
        return h2l_2D;
    }

    if(mesh_reader_verbose_output) pcout << "DOF_PER_FACE = " << dofs_per_face << std::endl;

    unsigned int next_index = 0;
    int subdegree = degree;
    int square_reduction = 0;
    while (subdegree > 0) {

        const unsigned int start = 0 + square_reduction;
        const unsigned int end = n - square_reduction;

        // First the four vertices
        {
            unsigned int i, j;
            // Bottom left
            i = start; j = start;
            h2l_2D[next_index++] = ij_to_num(i,j,n);
            // Bottom right
            i = end-1; j = start;
            h2l_2D[next_index++] = ij_to_num(i,j,n);
            // Top right
            i = end-1; j = end-1;
            h2l_2D[next_index++] = ij_to_num(i,j,n);
            // Top left
            i = start; j = end-1;
            h2l_2D[next_index++] = ij_to_num(i,j,n);
        }

        // Bottom line
        {
            unsigned int j = start;
            for (unsigned int i = start+1; i < end-1; ++i)
                h2l_2D[next_index++] = ij_to_num(i,j,n);
        }
        // Right line
        {
            unsigned int i = end-1;
            for (unsigned int j = start+1; j < end-1; ++j)
                h2l_2D[next_index++] = ij_to_num(i,j,n);
        }
        // Top line (right to left)
        {
            unsigned int j = end-1;
            //for (unsigned int i = start+1; i < end-1; ++i)
            // Need signed int otherwise, j=0 followed by --j results in j=UINT_MAX
            for (int i = end-2; i > (int)start; --i)
                h2l_2D[next_index++] = ij_to_num(i,j,n);
        }
        // Left line (top to bottom order)
        {
            unsigned int i = start;
            //for (unsigned int j = start+1; j < end-1; ++j)
            // Need signed int otherwise, j=0 followed by --j results in j=UINT_MAX
            for (int j = end-2; j > (int)start; --j)
                h2l_2D[next_index++] = ij_to_num(i,j,n);
        }

        subdegree -= 2;
        square_reduction += 1;

    }

    if (subdegree == 0) {
        const unsigned int middle = (n-1)/2;
        h2l_2D[next_index++] = ij_to_num(middle, middle, n);
    }

    Assert(next_index == dofs_per_face, dealii::ExcInternalError());

    return h2l_2D;
}

template <int dim>
std::vector<unsigned int>
gmsh_hierarchic_to_lexicographic(const unsigned int degree, const bool mesh_reader_verbose_output)
{

    /**
    * For Debug output
    */
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    // number of support points in each direction
    const unsigned int n = degree + 1;

    const unsigned int dofs_per_cell = dealii::Utilities::fixed_power<dim>(n);

    std::vector<unsigned int> h2l(dofs_per_cell);
    if (degree == 0) {
        h2l[0] = 0;
        return h2l;
    }

    // 
    const unsigned int inner_points_per_line = degree - 1;

    // The following lines of code are somewhat odd, due to the way the
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

            const unsigned int start = 0 + square_reduction;
            const unsigned int end = n - square_reduction;

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

        //INDEX START AT 0
        unsigned int next_index = 0;
        std::vector<unsigned int> face_position;
        std::vector<unsigned int> recursive_3D_position;
        std::vector<unsigned int> recursive_3D_nodes;
        std::vector<unsigned int> face_nodes;

        // First the eight corner vertices
        h2l[next_index++] = 0;                          // 0
        h2l[next_index++] = (1) * degree;               // 1
        h2l[next_index++] = (n + 1)*degree;             // 2
        h2l[next_index++] = (n) * degree;               // 3
        h2l[next_index++] = (n * n) * degree;           // 4
        h2l[next_index++] = (n * n + 1) * degree;       // 5
        h2l[next_index++] = (n * n + n + 1) * degree;   // 6
        h2l[next_index++] = (n * n + n) * degree;       // 7

        if (degree > 1) {

            //PHYSICAL INTERPRETATION OF THE NODES (MAPPING)
            //Degree is -2 because we are removing both end points
            face_nodes = face_node_finder(degree-2,mesh_reader_verbose_output);                     

           //For debug 
           if(mesh_reader_verbose_output) pcout << "DEGREE " << degree << std::endl;
           {
               unsigned int n = degree - 1;
               if(mesh_reader_verbose_output) pcout << "GMSH H2L " << std::endl;
               for (int j = n - 1; j >= 0; --j) {
                   for (unsigned int i = 0; i < n; ++i) {
                       const unsigned int ij = ij_to_num(i, j, n);
                       if(mesh_reader_verbose_output) pcout << face_nodes[ij] << " ";
                   }
                   if(mesh_reader_verbose_output) pcout << std::endl;
               }
           }

           if(mesh_reader_verbose_output) pcout << "" << std::endl;


            // line 0
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = i + 1;
               if(mesh_reader_verbose_output) pcout << "line 0 - " << i + 1 << std::endl;
            }

            // line 1
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (i + 1) * n;
               if(mesh_reader_verbose_output) pcout << "line 1 - " << (i + 1) * n << std::endl;
            }

            // line 2
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (i + 1) * n * n;
               if(mesh_reader_verbose_output) pcout << "line 2 - " << (i + 1) * n * n << std::endl;
            }

            // line 3
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (2 + i) * n - 1;
               if(mesh_reader_verbose_output) pcout << "line 3 - " << (2 + i) * n - 1 << std::endl;
            }

            // line 4
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (n * n * (i + 1)) + degree;
               if(mesh_reader_verbose_output) pcout << "line 4 - " << (n * n * (i + 1)) + degree << std::endl;
            }

            // line 5
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = n * n - (i + 1) - 1;
               if(mesh_reader_verbose_output) pcout << "line 5 - " << n * n - (i + 1) - 1 << std::endl;
            }

            // line 6
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (degree * n + (n * n * (i+1))) + degree;
               if(mesh_reader_verbose_output) pcout << "line 6 - " << (degree * n + (n * n * (i+1))) + degree << std::endl;
            }

            // line 7
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (degree * n + (n * n * (i+1)));
               if(mesh_reader_verbose_output) pcout << "line 7 - " << (degree * n + (n * n * (i+1))) << std::endl;
            }

            // line 8
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (n * n) * degree + (i + 1);
               if(mesh_reader_verbose_output) pcout << "line 8 - " << (n * n) * degree + (i + 1) << std::endl;
            }

            // line 9
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (i + 1) * n + (n * n) * degree;
               if(mesh_reader_verbose_output) pcout << "line 9 - " << (i + 1) * n + (n * n) * degree << std::endl;
            }
            // line 10
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = (2 + i) * n - 1 + (n * n) * degree;
               if(mesh_reader_verbose_output) pcout << "line 10 - " << (2 + i) * n - 1 + (n * n) * degree << std::endl;
            }
            // line 11
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                h2l[next_index++] = n * n * n - 1 - (i + 1);
               if(mesh_reader_verbose_output) pcout << "line 11 - " << n * n * n - 1 - (i + 1) << std::endl;
            }

            // inside quads
            // face 0
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
                   if(mesh_reader_verbose_output) pcout << "Face 0 : " << n + 1 + (n * j) + (i) << std::endl;
                    face_position.push_back(n + 1 + (n * j) + (i));
                }
            }

            for (unsigned int i = 0; i < (degree - 1) * (degree - 1); ++i) {
                h2l[next_index++] = face_position.at(face_nodes[i]);
            }

            face_position.clear();

            // face 1
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
                   if(mesh_reader_verbose_output) pcout << "Face 1 : " << (n * n * (i + 1)) + (j + 1) << std::endl;
                    face_position.push_back((n * n * (i + 1)) + (j + 1));
                }
            }

            for (unsigned int i = 0; i < (degree - 1) * (degree - 1); ++i) {
                h2l[next_index++] = face_position.at(face_nodes[i]);
            }

            face_position.clear();

            // face 2 -> Orientation is changed

            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
//                face_position.push_back((n * n * (i + 1)) + n + n * j);
                   if(mesh_reader_verbose_output) pcout << "Face 2 : " << (n * n * (j + 1)) + n + i * n << std::endl;
                    face_position.push_back((n * n * (j + 1)) + n + i * n);
                }
            }

            for (unsigned int i = 0; i < (degree - 1) * (degree - 1); ++i) {
                h2l[next_index++] = face_position.at(face_nodes[i]);
            }

            face_position.clear();

            // face 3

            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
                   if(mesh_reader_verbose_output) pcout << "Face 3 : " << n * (j + 2) - 1 + i * (n * n) + n * n << std::endl;
                    face_position.push_back(n * (j + 2) - 1 + i * (n * n) + n * n);
                }
            }

            for (unsigned int i = 0; i < (degree - 1) * (degree - 1); ++i) {
                h2l[next_index++] = face_position.at(face_nodes[i]);
            }

            face_position.clear();

            // face 4

            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
                   if(mesh_reader_verbose_output) pcout << "Face 4 : " << (n * n * i) + (n * n * 2) - (j + 1) - 1 << std::endl;
                    face_position.push_back((n * n * i) + (n * n * 2) - (j + 1) - 1);
                }
            }

            for (unsigned int i = 0; i < (degree - 1) * (degree - 1); ++i) {
                h2l[next_index++] = face_position.at(face_nodes[i]);
            }

            face_position.clear();

            // face 5

            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
                   if(mesh_reader_verbose_output) pcout << "Face 5 : " << (n * n * degree + n + (j + 1)) + i * n << std::endl;
                    face_position.push_back((n * n * degree + n + (j + 1)) + i * n);
                }
            }

            for (unsigned int i = 0; i < (degree - 1) * (degree - 1); ++i) {
                h2l[next_index++] = face_position.at(face_nodes[i]);
            }

            //Build the inner 3D structure with global index position
            for (unsigned int i = 0; i < inner_points_per_line; ++i) {
                for (unsigned int j = 0; j < inner_points_per_line; ++j) {
                    for (unsigned int k = 0; k < inner_points_per_line; ++k) {
//                        recursive_3D_position.push_back(n * n * (degree - i) - n - 2 - (n * k) - j);
                        recursive_3D_position.push_back(n * n + ((j+1) * n) + (k + 1) + (n * n * i));
                    }
                }
            }

           if(mesh_reader_verbose_output) pcout << "3D Inner Cube" << std::endl;
           for (unsigned int aInt: recursive_3D_position) {
               if(mesh_reader_verbose_output) pcout << aInt << std::endl;
           }

            /**
             * Now, we have an inside hex for hex of order 3 and more (2 has a single point)
             * Idea now is to use recursion and apply the same logic to the inner block
             */
            if (degree == 2) {
                h2l[next_index++] = recursive_3D_position.at(0);
            } else {

                /**
                 * Once this is out, we need to do some node processing, since the nodes are not at the correct locations
                 * Use the global index information to track it, i.e., get the transformed indices, and allocate them
                 * to the global index. This would make sense since the global index are always true for both GMSH and DEAL.II
                 */

               if(mesh_reader_verbose_output) pcout << "Begin recursive call to inner cube" << std::endl;
                recursive_3D_nodes = gmsh_hierarchic_to_lexicographic<dim>(degree - 2, mesh_reader_verbose_output);

               if(mesh_reader_verbose_output) pcout << "Printing recursive_3D_nodes" << std::endl;
               for (unsigned int aInt : recursive_3D_nodes) {
                   if(mesh_reader_verbose_output) pcout << aInt << std::endl;
               }

               if(mesh_reader_verbose_output) pcout << "Degree = " << degree << std::endl;
               if(mesh_reader_verbose_output) pcout << "Dim = " << dim << std::endl;

                //Apply the recursive_3D_nodes on the recursive_3D_position vector
                for (unsigned int i = 0; i < pow((degree - 1),3); ++i) {
                    h2l[next_index++] = recursive_3D_position.at(recursive_3D_nodes[i]);
                }
            }
        }

//        if(mesh_reader_verbose_output) pcout << "Next_index = " << next_index << std::endl;
//        Assert(next_index == dofs_per_cell, dealii::ExcInternalError());

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

/**
 * Function to get rotated indices in 2D
 **/ 
template <int dim, int spacedim>
bool get_new_rotated_indices(const dealii::CellAccessor<dim, spacedim>& cell,
                             const std::vector<dealii::Point<spacedim>>& all_vertices,
                             const std::vector<unsigned int>& deal_h2l,
                             const std::vector<unsigned int>& rotate_z90degree,
                             std::vector<unsigned int>& high_order_vertices_id)
{

    const unsigned int n_vertices = cell.n_vertices();
    for (int zr = 0; zr < 4; ++zr) {

        std::vector<char> matching(n_vertices);

        for (unsigned int i_vertex=0; i_vertex < n_vertices; ++i_vertex) {

            const unsigned int base_index = i_vertex;
            const unsigned int lexicographic_index = deal_h2l[base_index];

            const unsigned int vertex_id = high_order_vertices_id[lexicographic_index];
            const dealii::Point<dim,double> high_order_vertex = all_vertices[vertex_id];

            bool found = false;
            for (unsigned int i=0; i < n_vertices; ++i) {
                if (cell.vertex(i) == high_order_vertex) {
                    found = true;
                }
            }
            if (!found) {
                std::cout << "Wrong cell... High-order nodes do not match the cell's vertices." << std::endl;
                std::abort();
            }

            matching[i_vertex] = (high_order_vertex == cell.vertex(i_vertex)) ? 'T' : 'F';
        }

        bool all_matching = true;
        for (unsigned int i=0; i < n_vertices; ++i) {
            if (matching[i] == 'F') all_matching = false;
        }
        if (all_matching) return true;

        const auto high_order_vertices_id_temp = high_order_vertices_id;
        for (unsigned int i=0; i<high_order_vertices_id.size(); ++i) {
            high_order_vertices_id[i] = high_order_vertices_id_temp[rotate_z90degree[i]];
        }
    }
    return false;
}

/**
 * Function to get rotated indices in 3D
 **/ 
template <int dim, int spacedim>
bool get_new_rotated_indices_3D(const dealii::CellAccessor<dim, spacedim>& cell,
                                const std::vector<dealii::Point<spacedim>>& all_vertices,
                                const std::vector<unsigned int>& deal_h2l,
                                const std::vector<unsigned int>& rotate_x90degree_3D,
                                const std::vector<unsigned int>& rotate_y90degree_3D,
                                const std::vector<unsigned int>& rotate_z90degree_3D,
                                std::vector<unsigned int>& high_order_vertices_id_rotated,
                                const bool /*mesh_reader_verbose_output*/)
{

    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    //These variables are for 3D case
    auto high_order_flip_id_rotated = high_order_vertices_id_rotated;
    auto high_order_x_id_rotated = high_order_vertices_id_rotated;
    auto high_order_y_id_rotated = high_order_vertices_id_rotated;
    auto high_order_vertices_id = high_order_vertices_id_rotated;

    const unsigned int n_vertices = cell.n_vertices();

    bool good_rotation;

    for (int xr = 0; xr < 4; ++xr) {                    //Rotate in X-Axis                                                                       
        for (int zr = 0; zr < 4; ++zr) {                //Rotate in Y-Axis   
            for (int zr3d = 0; zr3d < 4; ++zr3d) {      //Rotate in Z-Axis

                std::vector<char> matching(n_vertices);

                //Parse through vertex points
                for (unsigned int i_vertex = 0; i_vertex < n_vertices; ++i_vertex) {                            

                    const unsigned int base_index = i_vertex;

                   //THESE ARE POSITIONS INDEX, SO GET THE LEXICOGRAHICAL_INDEX OF DEALII AND MAP IT BACK
                    const unsigned int lexicographic_index = deal_h2l[base_index];                                      

                    const unsigned int vertex_id = high_order_vertices_id_rotated[lexicographic_index];

                    //ALL_VERTICES IS IN HIERARCHICAL ORDER WITH POINTS (SO VERTEX_ID HITS BANG ON)
                    const dealii::Point<dim, double> high_order_vertex = all_vertices[vertex_id];                       

                    //.I.E. 0 4 20 24 5 -> POSITION 1 IS 4, SO FIRST NODE IN DEALII ORDERING WOULD BE AT POSITION 4 IN THE LEXICOGRPAHICAL ORDERING GENERATED BY GMSH

                    //THIS IS JUST TO SEE IF THE HIGHER ORDER NODES ARE FOUND (TECHNICALLY, WE SHOULD BE ONLY TARGETING 2D NODES, NOT THE 1D) 
                    //-> THIS SHOULD BE FILTERED OUT FROM HIGH_ORDER_VERTEX
                    bool found = false;
                    for (unsigned int i = 0; i < n_vertices; ++i) {
                        if (cell.vertex(i) == high_order_vertex) {
                            found = true;
                        }
                    }

                    if (!found) {
                        std::cout
                                << "Wrong cell... High order nodes do not match the cell's vertices | "
                                << "mpi_rank = " << mpi_rank << std::endl;
                        std::abort();
                    }

                    //CHECK FOR WHETHER THE VERTEX IS AT THE GOOD LOCATION OR NOT
                    matching[i_vertex] = (high_order_vertex == cell.vertex(i_vertex)) ? 0 : 1;
                }

                /**
                 * TECHNICALLY, THE MATCHING VECTOR SHOULD BE ALL 0 IF THEY ALL MATCH
                 */

                // if(mesh_reader_verbose_output) pcout << "********** CELL VERTEX **********" << std::endl;

                bool all_matching = true;
                for (unsigned int i = 0; i < n_vertices; ++i) {
                    if (matching[i] == 1) all_matching = false;
                }

                //Boolean for good rotation from the previous matching for loop
                good_rotation = all_matching;

                //If rotation is good, i.e., matching vertices, then we break.
                if (good_rotation) {
                    break;
                }

                //If we did not find a good rotation, we rotate in the Z-axis,
                //Swap the high_order_vertices_id_rotated by Z-axis rotation
                high_order_vertices_id = high_order_vertices_id_rotated;
                for (unsigned int i = 0; i < high_order_vertices_id.size(); ++i) {
                    high_order_vertices_id_rotated[i] = high_order_vertices_id[rotate_z90degree_3D[i]];
                }
            }

            //If rotation is good, i.e., matching vertices, then we break.
            if (good_rotation) {
                break;
            }

            //If we did not find a good rotation, we rotate in the Y-axis,
            //Swap the high_order_vertices_id_rotated by Y-axis rotation
            high_order_y_id_rotated = high_order_vertices_id_rotated;
            for (unsigned int i = 0; i < high_order_vertices_id.size(); ++i) {
                high_order_vertices_id_rotated[i] = high_order_y_id_rotated[rotate_y90degree_3D[i]];
            }
        }

        //If rotation is good, i.e., matching vertices, then we break.
        if (good_rotation) {
            break;
        }

        //If we did not find a good rotation, we rotate in the X-axis
        //Swap the high_order_vertices_id_rotated by X-axis rotation
        high_order_x_id_rotated = high_order_vertices_id_rotated;
        for (unsigned int i = 0; i < high_order_vertices_id.size(); ++i) {
            high_order_vertices_id_rotated[i] = high_order_x_id_rotated[rotate_x90degree_3D[i]];
        }

    }

    if (good_rotation) {
        return true;
    }

    return false;
}


// template <int dim, int spacedim>
// std::shared_ptr< HighOrderGrid<dim, double> >
// read_gmsh(std::string filename, bool periodic_x, bool periodic_y, bool periodic_z, int x_periodic_1, int x_periodic_2, int y_periodic_1, int y_periodic_2, int z_periodic_1, int z_periodic_2, true, int requested_grid_order)

template <int dim, int spacedim>
std::shared_ptr< HighOrderGrid<dim, double> >
read_gmsh(std::string filename, 
          const bool periodic_x, const bool periodic_y, const bool periodic_z, 
          const int x_periodic_1, const int x_periodic_2, 
          const int y_periodic_1, const int y_periodic_2, 
          const int z_periodic_1, const int z_periodic_2, 
          const bool mesh_reader_verbose_output,
          const bool do_renumber_dofs,
          int requested_grid_order,
          const bool use_mesh_smoothing)
{

    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

//    Assert(dim==2, dealii::ExcInternalError());
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
  
  
        // Read the end of the header and the first line of the nodes description
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
  
        // But the next thing should,
        // infile any case, be the list of
        // nodes:
        //AssertThrow(line == "$Nodes", PHiLiP::ExcInvalidGMSHInput(line));
    }
  
    std::vector<dealii::Point<spacedim>> vertices;

    // Set up mapping between numbering
    // infile msh-file (node) and infile the
    // vertices vector

    std::map<int, int> vertex_indices;
    read_gmsh_nodes( infile, vertices, vertex_indices, mesh_reader_verbose_output );
  
    // Assert we reached the end of the block
    infile >> line;
    static const std::string end_nodes_marker = "$EndNodes";
    //AssertThrow(line == end_nodes_marker, PHiLiP::ExcInvalidGMSHInput(line));
  
    // Now read infile next bit
    infile >> line;
    static const std::string begin_elements_marker = "$Elements";
    //AssertThrow(line == begin_elements_marker, PHiLiP::ExcInvalidGMSHInput(line));

    const unsigned int grid_order = find_grid_order<dim>(infile,mesh_reader_verbose_output);
  
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> triangulation;

    if(use_mesh_smoothing) {
        triangulation = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
    }
    else
    {
        triangulation = std::make_shared<Triangulation>(MPI_COMM_WORLD); // Dealii's default mesh smoothing flag is none. 
    }

    auto high_order_grid = std::make_shared<HighOrderGrid<dim, double>>(grid_order, triangulation);
  
    unsigned int n_entity_blocks, n_cells;
    int min_ele_tag, max_ele_tag;
    infile >> n_entity_blocks >> n_cells >> min_ele_tag >> max_ele_tag;

    // Set up array of p1_cells and subcells (faces). In 1d, there is currently no
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

        // For gmsh_file_format 4.1 the order of tag and dim is reversed,
        int tagEntity, dimEntity;
        infile >> dimEntity >> tagEntity >> cell_type >> numElements;
        material_id = tag_maps[dimEntity][tagEntity];

        const unsigned int cell_order = gmsh_cell_type_to_order(cell_type);

        unsigned int vertices_per_element = std::pow(2, dimEntity);
        unsigned int nodes_per_element = std::pow(cell_order + 1, dimEntity);


        for (unsigned int cell_per_entity = 0; cell_per_entity < numElements; ++cell_per_entity, ++global_cell) {

            // Ignore tag
            int tag;
            infile >> tag;

            if (dimEntity == dim) {

                /**
                 * When dimEntity == dim, this means we found a Face (2D) or Cell (3D)
                 **/

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

                // To make sure that the cast won't fail
                Assert(material_id <= std::numeric_limits<dealii::types::material_id>::max(),
                       dealii::ExcIndexRange( material_id, 0, std::numeric_limits<dealii::types::material_id>::max()));
                // We use only material_ids infile the range from 0 to dealii::numbers::invalid_material_id-1
                AssertIndexRange(material_id, dealii::numbers::invalid_material_id);

                p1_cells.back().material_id = material_id;

                // Transform from ucd to consecutive numbering
                for (unsigned int i = 0; i < vertices_per_element; ++i) {
                    //AssertThrow( vertex_indices.find(p1_cells.back().vertices[i]) != vertex_indices.end(),
                    //  dealii::ExcInvalidVertexIndexGmsh(global_cell, elm_number, p1_cells.back().vertices[i]));

                    // Vertex with this index exists
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

                // To make sure that the cast won't fail
                Assert(material_id <= std::numeric_limits<dealii::types::boundary_id>::max(),
                       dealii::ExcIndexRange( material_id, 0, std::numeric_limits<dealii::types::boundary_id>::max()));
                // We use only boundary_ids infile the range from 0 to dealii::numbers::internal_face_boundary_id-1
                AssertIndexRange(material_id, dealii::numbers::internal_face_boundary_id);

                subcelldata.boundary_lines.back().boundary_id = static_cast<dealii::types::boundary_id>(material_id);

                // Transform from ucd to consecutive numbering
                for (unsigned int &vertex : subcelldata.boundary_lines.back().vertices) {
                    if (vertex_indices.find(vertex) != vertex_indices.end()) {
                      vertex = vertex_indices[vertex];
                    } else {
                        // No such vertex index
                        //AssertThrow(false, dealii::ExcInvalidVertexIndex(cell_per_entity, vertex));
                        vertex = dealii::numbers::invalid_unsigned_int;
                        std::abort();
                    }
                }
            } else if (dimEntity == 2 && dimEntity < dim) {

                // Boundary info
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

                // To make sure that the cast won't fail
                Assert(material_id <= std::numeric_limits<dealii::types::boundary_id>::max(),
                       dealii::ExcIndexRange( material_id, 0, std::numeric_limits<dealii::types::boundary_id>::max()));
                // We use only boundary_ids infile the range from 0 to dealii::numbers::internal_face_boundary_id-1
                AssertIndexRange(material_id, dealii::numbers::internal_face_boundary_id);

                subcelldata.boundary_quads.back().boundary_id = static_cast<dealii::types::boundary_id>(material_id);

                // Transform from gmsh to consecutive numbering
                for (unsigned int &vertex : subcelldata.boundary_quads.back().vertices) {
                    if (vertex_indices.find(vertex) != vertex_indices.end()) {
                      vertex = vertex_indices[vertex];
                    } else {
                        // No such vertex index
                        //Assert(false, dealii::ExcInvalidVertexIndex(cell_per_entity, vertex));
                        vertex = dealii::numbers::invalid_unsigned_int;
                    }
                }
            } else if (cell_type == MSH_PNT) {
              // Read the indices of nodes given
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
  
    // Check that no forbidden arrays are used
    Assert(subcelldata.check_consistency(dim), dealii::ExcInternalError());

    AssertThrow(infile, dealii::ExcIO());

    // Check that we actually read some p1_cells.
    // AssertThrow(p1_cells.size() > 0, dealii::ExcGmshNoCellInformation());
  
    // Do some clean-up on vertices...
    const std::vector<dealii::Point<spacedim>> all_vertices = vertices;
    dealii::GridTools::delete_unused_vertices(vertices, p1_cells, subcelldata);

    // ... and p1_cells
    if (dim == spacedim) {
      dealii::GridReordering<dim, spacedim>::invert_all_cells_of_negative_grid(vertices, p1_cells);
    }
    dealii::GridReordering<dim, spacedim>::reorder_cells(p1_cells);
    triangulation->create_triangulation_compatibility(vertices, p1_cells, subcelldata); /// <<< FAILING HERE >>>

    triangulation->repartition();

    dealii::GridOut gridout;
    // gridout.write_mesh_per_processor_as_vtu(*(high_order_grid->triangulation), "tria");
  
    // in 1d, we also have to attach boundary ids to vertices, which does not
    // currently work through the call above
    if (dim == 1) {
        assign_1d_boundary_ids(boundary_ids_1d, *triangulation);
    }

    high_order_grid->initialize_with_triangulation_manifold();

    std::vector<unsigned int> deal_h2l = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_order);
    std::vector<unsigned int> deal_l2h = dealii::Utilities::invert_permutation(deal_h2l);

    std::vector<unsigned int> gmsh_h2l = gmsh_hierarchic_to_lexicographic<dim>(grid_order,mesh_reader_verbose_output);
    std::vector<unsigned int> gmsh_l2h = dealii::Utilities::invert_permutation(gmsh_h2l);

    int icell = 0;
    std::vector<dealii::types::global_dof_index> dof_indices(high_order_grid->fe_system.dofs_per_cell);

    std::vector<unsigned int> rotate_z90degree;

    /**
     * For 3D ROTATIONS (INITIATING ONCE HERE, SO WE DON'T RECOMPUTE FOR EACH ROTATION/CELL)
     */
    std::vector<unsigned int> rotate_z90degree_3D;
    std::vector<unsigned int> rotate_y90degree_3D;
    std::vector<unsigned int> rotate_x90degree_3D;
    std::vector<unsigned int> rotate_flip_z90degree_3D;

    //2D Rotation matrix (Pre allocate once)
    if constexpr(dim == 2) {
        if(mesh_reader_verbose_output) pcout << "Allocating 2D Rotate Z matrix..." << std::endl;
        rotate_indices<dim>(rotate_z90degree, grid_order+1, 'Z', mesh_reader_verbose_output);
    } else {
        //3D Rotation matrix (Pre allocate once)
        if(mesh_reader_verbose_output) pcout << "Allocating 3D Rotate Z matrix..." << std::endl;
        rotate_indices<dim>(rotate_z90degree_3D, grid_order + 1, 'Z', mesh_reader_verbose_output);
        if(mesh_reader_verbose_output) pcout << "Allocating 3D Rotate Y matrix..." << std::endl;
        rotate_indices<dim>(rotate_y90degree_3D, grid_order + 1, 'Y', mesh_reader_verbose_output);
        if(mesh_reader_verbose_output) pcout << "Allocating 3D Rotate X matrix..." << std::endl;
        rotate_indices<dim>(rotate_x90degree_3D, grid_order + 1, 'X', mesh_reader_verbose_output);
        if(mesh_reader_verbose_output) pcout << "Allocating 3D Rotate FLIP matrix..." << std::endl;
        rotate_indices<dim>(rotate_flip_z90degree_3D, grid_order + 1, 'F', mesh_reader_verbose_output);
    }

    if(mesh_reader_verbose_output) pcout << " " << std::endl;
    if(mesh_reader_verbose_output) pcout << "*********************************************************************\n";
    if(mesh_reader_verbose_output) pcout << "//********************** BEGIN ROTATING CELLS *********************//\n";
    if(mesh_reader_verbose_output) pcout << "*********************************************************************\n";
    if(mesh_reader_verbose_output) pcout << " " << std::endl;

    /**
     * Go through all cells and perform rotations to match gmsh with deal.ii
     */
    for (const auto &cell : high_order_grid->dof_handler_grid.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            auto &high_order_vertices_id = high_order_cells[icell].vertices;

            auto high_order_vertices_id_lexico = high_order_vertices_id;
            for (unsigned int ihierachic=0; ihierachic<high_order_vertices_id.size(); ++ihierachic) {
                const unsigned int lexico_id = gmsh_h2l[ihierachic];
                high_order_vertices_id_lexico[lexico_id] = high_order_vertices_id[ihierachic];
            }

            //auto high_order_vertices_id_rotated = high_order_cells[icell].vertices;
            auto high_order_vertices_id_rotated = high_order_vertices_id_lexico;

            if constexpr(dim == 2) {    //2D case

                bool good_rotation = get_new_rotated_indices(*cell, all_vertices, deal_h2l, rotate_z90degree, high_order_vertices_id_rotated);
                if (!good_rotation) {
                    //std::cout << "Couldn't find rotation... Flipping Z axis and doing it again" << std::endl;

                    // Flip Z-axis and do above again
                    std::vector<unsigned int> flipZ;
                    rotate_indices<dim>(flipZ, grid_order+1, '3', mesh_reader_verbose_output);
                    auto high_order_vertices_id_copy = high_order_vertices_id_rotated;
                    for (unsigned int i=0; i<high_order_vertices_id_rotated.size(); ++i) {
                        high_order_vertices_id_rotated[i] = high_order_vertices_id_copy[flipZ[i]];
                    }
                    good_rotation = get_new_rotated_indices(*cell, all_vertices, deal_h2l, rotate_z90degree, high_order_vertices_id_rotated);
                }

                if (!good_rotation) {
                    std::cout << "Couldn't find rotation after flipping either... Aborting..." << std::endl;
                    std::abort();
                }

            } else {    //3D case

                bool good_rotation = get_new_rotated_indices_3D(*cell, all_vertices, deal_h2l, rotate_x90degree_3D, rotate_y90degree_3D, rotate_z90degree_3D, high_order_vertices_id_rotated, mesh_reader_verbose_output);
                if (!good_rotation) {
                    if(mesh_reader_verbose_output) pcout << "3D -- Couldn't find rotation... Flipping Z axis and doing it again" << std::endl;

                    // Flip Z-axis and perform rotation again. 
                    auto high_order_vertices_id_copy = high_order_vertices_id_rotated;
                    for (unsigned int i=0; i<high_order_vertices_id_rotated.size(); ++i) {
                        high_order_vertices_id_rotated[i] = high_order_vertices_id_copy[rotate_flip_z90degree_3D[i]];
                    }

                    //Flip boolean should be included inside this boolean if statement
                    good_rotation = get_new_rotated_indices_3D(*cell, all_vertices, deal_h2l, rotate_x90degree_3D, rotate_y90degree_3D, rotate_z90degree_3D, high_order_vertices_id_rotated, mesh_reader_verbose_output);
                }

                if (!good_rotation) {
                    if(mesh_reader_verbose_output) pcout << "3D -- Couldn't find rotation after flipping 3D either... Aborting..." << std::endl;
                    std::abort();
                }
            }

            cell->get_dof_indices(dof_indices);
            for (unsigned int i_vertex = 0; i_vertex < high_order_vertices_id.size(); ++i_vertex) {

                const unsigned int base_index = i_vertex;
                const unsigned int lexicographic_index = deal_h2l[base_index];

                const unsigned int vertex_id = high_order_vertices_id_rotated[lexicographic_index];
                const dealii::Point<dim,double> vertex = all_vertices[vertex_id];


                for (int d = 0; d < dim; ++d) {
                    const unsigned int comp = d;
                    const unsigned int shape_index = high_order_grid->dof_handler_grid.get_fe().component_to_system_index(comp, base_index);
                    const unsigned int idof_global = dof_indices[shape_index];
                    high_order_grid->volume_nodes[idof_global] = vertex[d];
                }
            }
        }
        icell++;
    }

    if(mesh_reader_verbose_output) pcout << " " << std::endl;
    if(mesh_reader_verbose_output) pcout << "*********************************************************************\n";
    if(mesh_reader_verbose_output) pcout << "//********************** DONE ROTATING CELLS **********************//\n";
    if(mesh_reader_verbose_output) pcout << "*********************************************************************\n";
    if(mesh_reader_verbose_output) pcout << " " << std::endl;

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

        if(do_renumber_dofs) dealii::DoFRenumbering::Cuthill_McKee(dof_handler_equidistant);

        auto equidistant_nodes = high_order_grid->volume_nodes;
        equidistant_nodes.update_ghost_values();
        high_order_grid->volume_nodes.update_ghost_values();
        dealii::FETools::interpolate(dof_handler_equidistant, equidistant_nodes, high_order_grid->dof_handler_grid, high_order_grid->volume_nodes);
        high_order_grid->volume_nodes.update_ghost_values();
        high_order_grid->ensure_conforming_mesh();
    }

    high_order_grid->update_surface_nodes();
    high_order_grid->update_mapping_fe_field();
    high_order_grid->reset_initial_nodes();
    
    //Check for periodic boundary conditions and apply
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;

    if (periodic_x) {
        dealii::GridTools::collect_periodic_faces(*high_order_grid->triangulation, x_periodic_1, x_periodic_2, 0, matched_pairs);
    }

    if (periodic_y) {
        dealii::GridTools::collect_periodic_faces(*high_order_grid->triangulation, y_periodic_1, y_periodic_2, 1, matched_pairs);
    }

    if (periodic_z) {
        dealii::GridTools::collect_periodic_faces(*high_order_grid->triangulation, z_periodic_1, z_periodic_2, 2, matched_pairs);
    }

    if (periodic_x || periodic_y || periodic_z) {
        high_order_grid->triangulation->add_periodicity(matched_pairs);
    }

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

            if(do_renumber_dofs) dealii::DoFRenumbering::Cuthill_McKee(dof_handler_equidistant);

            auto equidistant_nodes = high_order_grid->volume_nodes;
            equidistant_nodes.update_ghost_values();
            grid->volume_nodes.update_ghost_values();
            dealii::FETools::interpolate(dof_handler_equidistant, equidistant_nodes, grid->dof_handler_grid, grid->volume_nodes);
            grid->volume_nodes.update_ghost_values();
            grid->ensure_conforming_mesh();
        }
        grid->update_surface_nodes();
        grid->update_mapping_fe_field();
        grid->reset_initial_nodes();

        //Check for periodic boundary conditions and apply
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        
        if (periodic_x) {
            dealii::GridTools::collect_periodic_faces(*grid->triangulation, x_periodic_1, x_periodic_2, 0, matched_pairs);
        }

        if (periodic_y) {
            dealii::GridTools::collect_periodic_faces(*grid->triangulation, y_periodic_1, y_periodic_2, 1, matched_pairs);
        }

        if (periodic_z) {
            dealii::GridTools::collect_periodic_faces(*grid->triangulation, z_periodic_1, z_periodic_2, 2, matched_pairs);
        }

        if (periodic_x || periodic_y || periodic_z) {
            grid->triangulation->add_periodicity(matched_pairs);
        }

        return grid;

    } else {
        return high_order_grid;
    }
}

template <int dim, int spacedim>
std::shared_ptr< HighOrderGrid<dim, double> >
read_gmsh(std::string filename, const bool do_renumber_dofs, int requested_grid_order, const bool use_mesh_smoothing)
{
  // default parameters
  const bool periodic_x = false;
  const bool periodic_y = false;
  const bool periodic_z = false;
  const int x_periodic_1 = 0; 
  const int x_periodic_2 = 0;
  const int y_periodic_1 = 0; 
  const int y_periodic_2 = 0;
  const int z_periodic_1 = 0; 
  const int z_periodic_2 = 0;
  const bool mesh_reader_verbose_output = true;

  return read_gmsh<dim,spacedim>(filename, 
    periodic_x, periodic_y, periodic_z, 
    x_periodic_1, x_periodic_2, 
    y_periodic_1, y_periodic_2, 
    z_periodic_1, z_periodic_2, 
    mesh_reader_verbose_output,
    do_renumber_dofs,
    requested_grid_order,
    use_mesh_smoothing);
}

#if PHILIP_DIM!=1 
template std::shared_ptr< HighOrderGrid<PHILIP_DIM, double> > read_gmsh<PHILIP_DIM,PHILIP_DIM>(std::string filename, const bool periodic_x, const bool periodic_y, const bool periodic_z, const int x_periodic_1, const int x_periodic_2, const int y_periodic_1, const int y_periodic_2, const int z_periodic_1, const int z_periodic_2, const bool mesh_reader_verbose_output, const bool do_renumber_dofs, int requested_grid_order, const bool use_mesh_smoothing);
template std::shared_ptr< HighOrderGrid<PHILIP_DIM, double> > read_gmsh<PHILIP_DIM,PHILIP_DIM>(std::string filename, const bool do_renumber_dofs, int requested_grid_order, const bool use_mesh_smoothing);
#endif

} // namespace PHiLiP
