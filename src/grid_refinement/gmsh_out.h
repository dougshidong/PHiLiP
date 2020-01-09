#ifndef __GMSH_OUT_H__
#define __GMSH_OUT_H__

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

namespace PHiLiP {

namespace GridRefinement {

// wrapper for the set of functions output to gmsh through pos
template <int dim, typename real>
class GmshOut
{
public:
    // writing the .pos file for use with gmsh
    static void write_pos(
        const dealii::Triangulation<dim, dim> &tria,
        dealii::Vector<real>                   data,
        std::ostream &                         out);

    // writing the geo file
    static void write_geo(
        std::vector<std::string> &posFile_vec,
        std::ostream &            out);

private:
    // writing the part of the geo file for a hyper cube
    static void write_geo_hyper_cube(
        const double  left,
        const double  right,
        std::ostream &out);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GMSH_OUT_H__
