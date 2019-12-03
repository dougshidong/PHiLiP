#include <float.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

#include "gmsh_out.h"

namespace PHiLiP {

namespace GridRefinement {

/* Set of functions for outputting necessary data to pass to GMSH mesh generators
 * Need 4 sets of output types
 * 1D:
 *      SL - Scalar Line
 *      SP - Scalar Point
 *      VL - Vector Line
 *      VP - Vector Point
 * 2D:
 *      SQ - Scalar Quad
 *      SP - Scalar Point
 *      VQ - Vector Quad
 *      VP - Vector Point
 * 3D:
 *      SH - Scalar Hex
 *      SP - Scalar Point
 *      VH - Vector Hex
 *      VP - Vector Point
 * 
 * Control this output based on the same general form as used in DEALII::DataOutBase
 */

// file for writing SQ output 

// also need some system of writing the data vectors, preferably identically to how
// the usual data_out as used in dg does it
// probably needs to add a flag system for the quad/point decision

// might need to add a dofhandler when processing the data
template <int dim, typename real>
void GmshOut<dim,real>::write_pos(
    const dealii::Triangulation<dim, dim> &tria,
    dealii::Vector<real>                   data,
    std::ostream &                         out)
{
    // get positions
    // const std::vector<dealii::Point<dim>> &vertices    = tria.get_vertices();
    // const std::vector<bool> &                   vertex_used = tria.get_used_vertices();

    const unsigned int n_vertices = tria.n_used_vertices();

    typename dealii::Triangulation<dim, dim>::active_cell_iterator cell =
        tria.begin_active();
    const typename dealii::Triangulation<dim, dim>::active_cell_iterator endc =
        tria.end();

    // write header
    out << "/*********************************** " << '\n'
        << " * BACKGROUND MESH FIELD GENERATED * " << '\n'
        << " * AUTOMATICALLY BY PHiLiP LIBRARY * " << '\n'
        << " ***********************************/" << '\n';

    // number of vertices
    out << "// File contains n_vertices = " << n_vertices << '\n' << '\n';

    // write the "view" header
    out << "View \"background mesh\" {"  << '\n';

    // writing the main body
    // SQ(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4){v1,v2,v3,v4};
    for(cell = tria.begin_active(); cell!=endc; ++cell){
        if(!cell->is_locally_owned()) continue;

        // select the output type
        if(dim == 2)
            out << "SQ(";
        if(dim == 3)
            out << "SH(";

        // writing the coordinates
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
        {
            // Fix for the difference in numbering orders (CCW)
            // DEALII: 2D=[[0,1],[2,3]], 3D=[[[0,1],[2,3]],[[4,5],[6,7]]]
            // GMSH:   2D=[[0,1],[3,2]], 3D=[[[0,1],[3,2]],[[4,5],[7,6]]]
            dealii::Point<dim> pos;
            if((vertex+2)%4 == 0){ // (2,6) -> (3,7)
                pos = cell->vertex(vertex+1);
            }else if((vertex+1)%4 == 0){ // (3,7) -> (2,6)
                pos = cell->vertex(vertex-1);
            }else{
                pos = cell->vertex(vertex);
            }

            if(vertex != 0){out << ",";} 

            // x
            if(dim >= 1){out << pos[0] << ",";}
            else        {out << 0      << ",";}
            // y
            if(dim >= 2){out << pos[1] << ",";}
            else        {out << 0      << ",";}
            // z
            if(dim >= 3){out << pos[2];}
            else        {out << 0;}
        }

        out << "){";

        // writing the data values
        // for now just putting 1.0
        // out << "1.0,2.0,3.0,4.0";
        // for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
        // {
        //     // Fix for the difference in numbering orders (CCW)
        //     // DEALII: 2D=[[0,1],[2,3]], 3D=[[[0,1],[2,3]],[[4,5],[6,7]]]
        //     // GMSH:   2D=[[0,1],[3,2]], 3D=[[[0,1],[3,2]],[[4,5],[7,6]]]
        //     dealii::Point<dim> pos;
        //     if((vertex+2)%4 == 0){ // (2,6) -> (3,7)
        //         pos = cell->vertex(vertex+1);
        //     }else if((vertex+1)%4 == 0){ // (3,7) -> (2,6)
        //         pos = cell->vertex(vertex-1);
        //     }else{
        //         pos = cell->vertex(vertex);
        //     }
        //     // just going to assume the dim is 2 for now
        //     if(vertex != 0){out << ",";} 
        //     if(dim == 2){
        //         double x = pos[0], y = pos[1];
        //         double v = x*y+0.01;
        //         out<<v;
        //     }
        // }

        // getting the cellwise value
        real v = data[cell->active_cell_index()];
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
        {
            if(vertex != 0){out << ",";}
            out << v;
        }

        out << "};" << '\n';
    }

    // write the "view footer"
    out << "}; //View \"background mesh\" " << '\n';

    out << std::flush;
}

// writing the geo file which calls the meshing
// need to add a set of flags here describing what to write
template <int dim, typename real>
void GmshOut<dim,real>::write_geo(
    std::string   posFile,
    std::ostream &out)
{
    // write header
    out << "/*********************************** " << '\n'
        << " * MESH GEOMETRY FILE GENERATED    * " << '\n'
        << " * AUTOMATICALLY BY PHiLiP LIBRARY * " << '\n'
        << " ***********************************/" << '\n' << '\n';

    // writing the geometry of the part
    // TODO: read from a parameter list what shape (could be CAD)
    write_geo_hyper_cube(0.0, 1.0, out);

    // writing the information about the recombination
    // TODO: read from a parameter list what schemes
    out << "// Merging the background mesh and recombine" << '\n';
    out << "Merge \"" << posFile << "\";" << '\n';

    // this line may be dependent on the name in the other file
    out << "Background Mesh View[0];" << '\n';

    // always using surface{1}
    out << "Recombine Surface{1};" << '\n';

    // default is the advancing delquad
    // TODO: add parameter file for other options (and 3d)
    out << "Mesh.Algorithm = 8;" << '\n';

    out << "Mesh.SubdivisionAlgorithm = 1;" << '\n'; 

    out << std::flush;
}

// writes the geometry of a hypercube to the file as the domain to be meshed
// TODO: Colorize the boundary
template <int dim, typename real>
void GmshOut<dim,real>::write_geo_hyper_cube(
    const double  left,
    const double  right,
    std::ostream &out)
{
    // placing the poins at each of the corners (can be in any order)
    // splitting the dimensional cases for easier reading
    
    if(dim == 2){
        out << "// Points" << '\n'; 
        out << "Point(1)={" << left  << "," << left  << "," << 0 << "};" << '\n';
        out << "Point(2)={" << right << "," << left  << "," << 0 << "};" << '\n';
        out << "Point(3)={" << right << "," << right << "," << 0 << "};" << '\n';
        out << "Point(4)={" << left  << "," << right << "," << 0 << "};" << '\n';
        out << '\n';

        out << "// Lines" << '\n';
        out << "Line(1)={1,2};" << '\n';
        out << "Line(2)={2,3};" << '\n';
        out << "Line(3)={3,4};" << '\n';
        out << "Line(4)={4,1};" << '\n';
        out << '\n';

        out << "// Loop" <<  '\n';
        out << "Line Loop(1)={1,2,3,4};" << '\n';
        out << '\n';

        out << "// Surface" << '\n';
        out << "Plane Surface(1) = {1};" << '\n';
        out << '\n';
    }

}

template class GmshOut <PHILIP_DIM, double>;
template class GmshOut <PHILIP_DIM, float>;

} // namespace GridRefinement

} //namespace PHiLiP