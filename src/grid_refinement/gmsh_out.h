#ifndef __GMSH_OUT_H__
#define __GMSH_OUT_H__

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

namespace PHiLiP {

namespace GridRefinement {

// wrapper for the set of functions output to gmsh through pos
/// GmshOut class
/** This class contains static methods for writing files neeed to 
  * describe the domain geometry and adapted size or anisotropic
  * tesnor fields compatible for use with standard GMSH interfaces
  * for use with quad remeshing.
  * 
  * This class also contains a utility function to call GMSH from 
  * the console if it has been enabled with ENABLE_GMSH=1 in the 
  * CMakeLists.txt with suitable path to GMSH also specified.
  * 
  * GMSH is an open source 2D and 3D mesh generation software with
  * specialized functions for treatment of all-quad meshing domains
  * (E.g. \f$L^\infty\f$ advacing front methods and BlossomQuad for recombination).
  * It uses a combination of .geo files to describe the part and case setup
  * and .pos files for addittional field information (e.g. mesh size).
  * See link for more details: https://gmsh.info/
  * 
  * Note: Currently only supported in 2D
  */ 
template <int dim, typename real>
class GmshOut
{
public:
    /// Write scalar .pos file for use with GMSH
    /** Write SQ (Scalar Quad) output based on input triangulation (the original mesh)
      * and a size field stored in vector form. The specified format is written to the ostream
      * which can be an output file (or console for debugging).
      */ 
    static void write_pos(
        const dealii::Triangulation<dim,dim> &tria,
        dealii::Vector<real>                  data,
        std::ostream &                        out);

    /// Write anisotropic tensor .pos file for use with GMSH
    /** Written as TT (Tensor triangles) using a split quad representation.
      * Each element is described by a 3x3 matrix specifying the local anisotropic
      * quadratic sizing function \f$x^T \mathcal{M} x\f$ for use with the BAMG meshing
      * methods in 2D. Use with write_geo_anisotropic to specify recombination with
      * default BlossomQuad methods for best perfomance.
      */ 
    static void write_pos_anisotropic(
        const dealii::Triangulation<dim,dim>&                   tria,
        const std::vector<dealii::SymmetricTensor<2,dim,real>>& data,
        std::ostream&                                           out,
        const int                                               p_scale = 1);

    /// Writes the central .geo file for call to GMSH on main process with isotropic quad meshing
    /** posFile_vec contains list of data files written from each subprocess to the filesystem 
      * to be read from parralel run for serial remeshing. Uses advancing front delaunay method 
      * for quads based on \f$L^\infty\f$ node insertion to produce a right angle mesh. Final isotropic
      * mesh with target size field is reocmbined using BlossomQuad to produce all-quad output mesh
      * Note: Currently only hybercube geometries are supported
      */ 
    static void write_geo(
        std::vector<std::string> &posFile_vec,
        std::ostream &            out);

     /// Writes the central .geo file for call to GMSH on main process with anisotropic quad meshing
    /** posFile_vec contains list of data files written from each subprocess to the filesystem 
      * to be read from parralel run for serial remeshing. Uses BAMG for anisotropic triangular mesh
      * generation with BlossomQuad recombination to all-quad output mesh (may fail in some versions).
      * Note: Currently only hybercube geometries are supported
      */ 
    static void write_geo_anisotropic(
        std::vector<std::string> &posFile_vec,
        std::ostream &            out);

    /// Performs command line call to GMSH for grid generation (if availible)
    /** Read input .geo and writes the final .msh to specified output.
     * Requires that ENABLE_GMSH has been set to 1 and the GMSH_PATH has been
     * defined in the CMakeLists.txt file. Returns 1 if the system call is 
     * performed, 0 otherwise.
      */ 
    static int call_gmsh(
        std::string geo_name,
        std::string output_name);

private:
    /// Writes the part of the .geo file associated wit the hyperdube geometry
    /** left and right specify the min and max axis aligned values in each direction.
      * Use colorize to number the domain boundaries in GMSH form
      */
    static void write_geo_hyper_cube(
        const double  left,
        const double  right,
        std::ostream &out,
        const bool    colorize = true);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GMSH_OUT_H__
