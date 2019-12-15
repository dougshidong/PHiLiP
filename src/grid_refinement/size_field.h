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

    // computes isotropic size field (h has only 1 component) for a non-uniform p_field
    static void isotropic_h(
        real                                 complexity,            // (input) complexity target
        dealii::Vector<real> &               B,                     // only one since p is constant
        dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        dealii::Vector<real> &               h_field,               // (output) size field
        dealii::Vector<real> &               p_field);              // (input)  poly field

    // computes updated p-field with a constant h-field
    static void isotropic_p(
        dealii::Vector<real> &               Bm,                    // constant for p-1
        dealii::Vector<real> &               B,                     // constant for p
        dealii::Vector<real> &               Bp,                    // constant for p+1
        dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        dealii::Vector<real> &               h_field,               // (input) size field
        dealii::Vector<real> &               p_field);              // (output) poly field

    // updates both the h-field and p-field
    static void isotropic_hp(
        real                                 complexity,            // target complexity
        dealii::Vector<real> &               Bm,                    // constant for p-1
        dealii::Vector<real> &               B,                     // constant for p
        dealii::Vector<real> &               Bp,                    // constant for p+1
        dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        dealii::Vector<real> &               h_field,               // (output) size field
        dealii::Vector<real> &               p_field);              // (output) poly field

protected:
    // given a p-field, redistribute h_field according to B
    static void update_h_optimal(
        real                          lambda,      // (input) bisection parameter
        dealii::Vector<real> &        B,           // constant for current p
        dealii::hp::DoFHandler<dim> & dof_handler, // dof_handler
        dealii::Vector<real> &        h_field,     // (output) size field
        dealii::Vector<real> &        p_field);    // (input)  poly field

    static real evaluate_complexity(
        dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        dealii::Vector<real> &               h_field,               // (input) size field
        dealii::Vector<real> &               p_field);              // (input) poly field  

    static real bisection(
        std::function<real(real)> func,         // lambda function that takes real -> real 
        real                      lower_bound,  // lower bound of the search
        real                      upper_bound); // upper bound of the search

};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __SIZE_FIELD_H__
