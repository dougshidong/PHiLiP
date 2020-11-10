#ifndef __GMSH_OUT_H__
#define __GMSH_OUT_H__

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

namespace PHiLiP {

namespace GridRefinement {

// wrapper for the set of functions output to gmsh through pos
template <int dim, typename real>
class GmshOut
{
public:
    // writing the .pos file for use with gmsh
    static void write_pos(
        const dealii::Triangulation<dim,dim> &tria,
        dealii::Vector<real>                  data,
        std::ostream &                        out);

    // writing anisotropic (tensor based) .pos file for use with gmsh
    static void write_pos_anisotropic(
        const dealii::Triangulation<dim,dim>&                   tria,
        const std::vector<dealii::SymmetricTensor<2,dim,real>>& data,
        std::ostream&                                           out);

    // writing the geo file
    static void write_geo(
        std::vector<std::string> &posFile_vec,
        std::ostream &            out);

    // writing the (anisotropic) geo file
    static void write_geo_anisotropic(
        std::vector<std::string> &posFile_vec,
        std::ostream &            out);

private:
    // writing the part of the geo file for a hyper cube
    static void write_geo_hyper_cube(
        const double  left,
        const double  right,
        std::ostream &out,
        const bool    colorize = true);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GMSH_OUT_H__
