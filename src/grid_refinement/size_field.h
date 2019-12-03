#ifndef __SIZE_FIELD_H__
#define __SIZE_FIELD_H__

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

namespace PHiLiP {

namespace GridRefinement {

    // wrapper for the set of functions output to gmsh through pos
    template <int dim, typename real>
    class SizeField
    {
    public:
        // computes the isotropic size field (h has only 1 component) for a uniform (p-dist constant) input
        static void isotropic_uniform(
            const dealii::Triangulation<dim, dim> &                   tria,                             // triangulation
            const dealii::Mapping<dim, dim > &                        mapping,                          // mapping field used in computed JxW
            const dealii::FiniteElement<dim, dim> &                   fe,                               // for fe_values integration, assumed constant for now
            std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,   // manufactured solution
            real                                                      complexity,                       // continuous dof measure
            dealii::Vector<real> &                                    h_field);           
    };

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __SIZE_FIELD_H__
