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
        const real &                               complexity,   // (input) complexity target
        const dealii::Vector<real> &               B,            // only one since p is constant
        const dealii::hp::DoFHandler<dim> &        dof_handler,  // dof_handler
        dealii::Vector<real> &                     h_field,      // (output) size field
        const real &                               poly_degree); // (input)  polynomial degree

    // computes isotropic size field (h has only 1 component) for a non-uniform p_field
    static void isotropic_h(
        const real                                 complexity,            // (input) complexity target
        const dealii::Vector<real> &               B,                     // only one since p is constant
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        dealii::Vector<real> &                     h_field,               // (output) size field
        const dealii::Vector<real> &               p_field);              // (input)  poly field

    // computes updated p-field with a constant h-field
    static void isotropic_p(
        const dealii::Vector<real> &               Bm,                    // constant for p-1
        const dealii::Vector<real> &               B,                     // constant for p
        const dealii::Vector<real> &               Bp,                    // constant for p+1
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        const dealii::Vector<real> &               h_field,               // (input) size field
        dealii::Vector<real> &                     p_field);              // (output) poly field

    // updates both the h-field and p-field
    static void isotropic_hp(
        const real                                 complexity,            // target complexity
        const dealii::Vector<real> &               Bm,                    // constant for p-1
        const dealii::Vector<real> &               B,                     // constant for p
        const dealii::Vector<real> &               Bp,                    // constant for p+1
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        dealii::Vector<real> &                     h_field,               // (output) size field
        dealii::Vector<real> &                     p_field);              // (output) poly field

protected:
    // given a p-field, redistribute h_field according to B
    static void update_h_optimal(
        const real                          lambda,      // (input) bisection parameter
        const dealii::Vector<real> &        B,           // constant for current p
        const dealii::hp::DoFHandler<dim> & dof_handler, // dof_handler
        dealii::Vector<real> &              h_field,     // (output) size field
        const dealii::Vector<real> &        p_field);    // (input)  poly field

    static real evaluate_complexity(
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        const dealii::Vector<real> &               h_field,               // (input) size field
        const dealii::Vector<real> &               p_field);              // (input) poly field  

    static real bisection(
        const std::function<real(real)> func,         // lambda function that takes real -> real 
        real                            lower_bound,  // lower bound of the search
        real                            upper_bound); // upper bound of the search

};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __SIZE_FIELD_H__
