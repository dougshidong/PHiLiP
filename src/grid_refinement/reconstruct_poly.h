
#ifndef __RECONSTRUCT_POLY_H__
#define __RECONSTRUCT_POLY_H__

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

namespace PHiLiP {

namespace GridRefinement {

template <int dim, typename real>
class ReconstructPoly
{
public:
    static void reconstruct_directional_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
        const dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
        const unsigned int&                                    rel_order,             // order of the reconstruction
        std::vector<dealii::Tensor<1,dim,real>>&               A);                     // (output) holds the largest (scaled) derivative in each direction and then in each orthogonal plane

private:
    static std::array<unsigned int, dim> compute_index(
        const unsigned int i,
        const unsigned int size);

    template <typename DoFCellAccessorType>
    static dealii::Vector<real> reconstruct_H1_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution,
        const dealii::hp::MappingCollection<dim> &              mapping_collection,
        const dealii::hp::FECollection<dim> &                   fe_collection,
        const dealii::hp::QCollection<dim> &                    quadrature_collection,
        const dealii::UpdateFlags &                             update_flags);

    template <typename DoFCellAccessorType>
    static dealii::Vector<real> reconstruct_L2_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution,
        const dealii::hp::MappingCollection<dim> &              mapping_collection,
        const dealii::hp::FECollection<dim> &                   fe_collection,
        const dealii::hp::QCollection<dim> &                    quadrature_collection,
        const dealii::UpdateFlags &                             update_flags);

    template <typename DoFCellAccessorType>
    static std::vector<DoFCellAccessorType> get_patch_around_dof_cell(
        const DoFCellAccessorType &cell);

};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __RECONSTRUCT_POLY_H__
