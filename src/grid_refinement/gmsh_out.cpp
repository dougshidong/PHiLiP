#include <float.h>
#include <string>

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

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
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
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
        real scale = 1; // no subdivision
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
            if(vertex != 0){out << ",";}
            out << v*scale; 
        }

        out << "};" << '\n';
    }

    // write the "view footer"
    out << "}; //View \"background mesh\" " << '\n';

    out << std::flush;
}

// writing anisotropic (tensor based) .pos file for use with gmsh
template <int dim, typename real>
void GmshOut<dim,real>::write_pos_anisotropic(
    const dealii::Triangulation<dim,dim>&                   tria,
    const std::vector<dealii::SymmetricTensor<2,dim,real>>& data,
    std::ostream&                                           out,
    const int                                               p_scale)
{
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
    // TT(x1,y1,z1,x2,y2,z2,x3,y3,z3){M1, M2, M3};
    // where Mi = [m11, m12, m13, m21, m22, m23, m31, m32, m33];
    for(cell = tria.begin_active(); cell!=endc; ++cell){
        if(!cell->is_locally_owned()) continue;

        // cell-wise value
        const dealii::Tensor<2,dim,real> val = data[cell->active_cell_index()];

        // only implemented for dim == 2 currently
        if(dim == 2){

            // writing the coordinates
            std::vector<dealii::Point<dim>> vertices;
            vertices.reserve(dealii::GeometryInfo<dim>::vertices_per_cell);
            for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
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

                vertices.push_back(pos);
            }

            // splitting the quad into two triangles and output
            // forming two triangels [0,3,2] and [0,2,1]
            for(const auto& tri: 
                std::array<std::array<int,3>,2>{{ 
                    {{0,3,2}}, 
                    {{0,2,1}} }}){
                // writing the coordinates
                out << "TT(";
                for(unsigned int i = 0; i < tri.size(); ++i){
                    if(i != 0){out << ",";}

                    // x
                    dealii::Point<dim> pos = vertices[tri[i]];
                    if(dim >= 1){out << pos[0] << ",";}
                    else        {out << 0      << ",";}
                    // y
                    if(dim >= 2){out << pos[1] << ",";}
                    else        {out << 0      << ",";}
                    // z
                    if(dim >= 3){out << pos[2];}
                    else        {out << 0;}
                }

                // writing the tensor data (for each node)
                out << "){";
                bool flag = false;
                for(unsigned int vertex = 0; vertex < tri.size(); ++vertex){
                    
                    // writing the tensor itself, always 3x3
                    const unsigned int N = 3;

                    // empirical scaling for the complexity match
                    double scale = 1.0;
                    if(p_scale == 1){
                        scale = 0.25*0.5/sqrt(2.0);
                    }else if(p_scale == 2){
                        scale = 0.25/sqrt(3.0);
                    }else if(p_scale == 3){
                        scale = 0.25/sqrt(2.0);
                    }

                    for(unsigned int i = 0; i < N; ++i){
                        for(unsigned int j = 0; j < N; ++j){

                            // only skipping comma on first point
                            if(flag){
                                out << ",";
                            }else{
                                flag  = true;
                            }

                            // data only specified for upper dim x dim
                            if( (i < dim) && (j < dim) ){
                                out << scale*val[i][j];
                            }else{
                                // adding 1.0 along diagonal
                                if( i == j ){
                                    out << "1.0";
                                }else{
                                    out << "0";
                                }
                            }

                        }
                    }
                }

                out << "};" << '\n';

            }

        }
    }

    // write the "view footer"
    out << "}; //View \"background mesh\" " << '\n';
    out << std::flush;

}

// writing the geo file which calls the meshing
// need to add a set of flags here describing what to write
template <int dim, typename real>
void GmshOut<dim,real>::write_geo(
    std::vector<std::string> &posFile_vec,
    std::ostream &            out)
{
    // write header
    out << "/*********************************** " << '\n'
        << " * MESH GEOMETRY FILE GENERATED    * " << '\n'
        << " * AUTOMATICALLY BY PHiLiP LIBRARY * " << '\n'
        << " ***********************************/" << '\n' << '\n';

    // the background field should fully specify the mesh size, points seem to fix some skewness
    out << "Mesh.CharacteristicLengthFromPoints = 1;" << '\n'
        << "Mesh.CharacteristicLengthFromCurvature = 0;" << '\n'
        << "Mesh.CharacteristicLengthExtendFromBoundary = 0;" << '\n' << '\n';

    // default is the advancing delquad
    // TODO: add parameter file for other options (and 3d)
    out << "Mesh.Algorithm = 8;" << '\n'
        << "Mesh.RecombinationAlgorithm = 3;" << '\n' 
        << "Mesh.RecombineAll = 1;" << '\n' << '\n'; 

    // writing the geometry of the part
    // TODO: read from a parameter list what shape (could be CAD)
    write_geo_hyper_cube(0.0, 1.0, out);

    // writing the information about the recombination
    // TODO: read from a parameter list what schemes
    out << "// Merging the background mesh and recombine" << '\n';
    for(unsigned int ifile = 0; ifile < posFile_vec.size(); ++ifile)
        out << "Merge \"" << posFile_vec[ifile] << "\";" << '\n';

    // combining the views
    out << "Combine Views;" << '\n';

    // this line may be dependent on the name in the other file
    out << "Background Mesh View[0];" << '\n';

    out << std::flush;
}

// writing the (anisotropic) geo file
template <int dim, typename real>
void GmshOut<dim,real>::write_geo_anisotropic(
    std::vector<std::string> &posFile_vec,
    std::ostream &            out)
{
    // write header
    out << "/*********************************** " << '\n'
        << " * MESH GEOMETRY FILE GENERATED    * " << '\n'
        << " * AUTOMATICALLY BY PHiLiP LIBRARY * " << '\n'
        << " ***********************************/" << '\n' << '\n';

    // the background field should fully specify the mesh size, points seem to fix some skewness
    out << "Mesh.CharacteristicLengthFromPoints = 1;" << '\n'
        << "Mesh.CharacteristicLengthFromCurvature = 0;" << '\n'
        << "Mesh.CharacteristicLengthExtendFromBoundary = 0;" << '\n' << '\n';

    // default is the BAMG for anisotropy
    out << "Mesh.Algorithm = 7;" << '\n'
        // << "Mesh.SmoothRatio = 1.5;" << '\n'
        << "Mesh.RecombinationAlgorithm = 2;" << '\n' 
        << "Mesh.RecombineAll = 1;" << '\n' << '\n'; 

    // writing the geometry of the part
    // TODO: read from a parameter list what shape (could be CAD)
    write_geo_hyper_cube(0.0, 1.0, out);

    // writing the information about the recombination
    // TODO: read from a parameter list what schemes
    out << "// Merging the background mesh and recombine" << '\n';
    for(unsigned int ifile = 0; ifile < posFile_vec.size(); ++ifile)
        out << "Merge \"" << posFile_vec[ifile] << "\";" << '\n';

    // combining the views
    out << "Combine Views;" << '\n';

    // this line may be dependent on the name in the other file
    out << "Background Mesh View[0];" << '\n';

    out << std::flush;
}

// writes the geometry of a hypercube to the file as the domain to be meshed
template <int dim, typename real>
void GmshOut<dim,real>::write_geo_hyper_cube(
    const double  left,
    const double  right,
    std::ostream &out,
    const bool    colorize)
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

        // colorizes the boundary in the style of DEALII internal numbering
        if(colorize){
            out << "// Colorize" << '\n';
            out << "Physical Curve (2) = {1};" << '\n';
            out << "Physical Curve (1) = {2};" << '\n';
            out << "Physical Curve (3) = {3};" << '\n';
            out << "Physical Curve (0) = {4};" << '\n';
            out << '\n';
        }
    }

}

template <int dim, typename real>
int GmshOut<dim,real>::call_gmsh(
    std::string geo_name,
    std::string output_name)
{
#if ENABLE_GMSH && defined GMSH_PATH
    // enabled, call with suitable args
    std::string args = " " + geo_name + " -" + std::to_string(dim) + " -save_all -o " + output_name;
    std::string cmd = GMSH_PATH + args;
    std::cout << "Command is: " << cmd << '\n';
    int ret = std::system(cmd.c_str());
    (void) ret;
    return 1;
#else
    // disabled
    (void) geo_name;
    (void) output_name;
    std::cerr << "Error: Call to gmsh without gmsh enabled." << std::endl;
    std::cerr << "       Please set ENABLE_GMSH and GMSH_PATH and try again." << std::endl;
    return 0;
#endif
}


template class GmshOut <PHILIP_DIM, double>;
template class GmshOut <PHILIP_DIM, float>;

} // namespace GridRefinement

} //namespace PHiLiP