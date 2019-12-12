
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
    void reconstruct_directional_derivative(
        dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
        dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
        dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
        dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
        dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
        dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
        unsigned int                                     rel_order,             // order of the reconstruction
        dealii::Vector<dealii::Tensor<1,dim,real>>&      A);                    // (output) holds the largest (scaled) derivative in each direction and then in each orthogonal plane

private:
    static std::array<unsigned int, dim> compute_index(
        const unsigned int i,
        const unsigned int size);

    template <typename DoFCellAccessorType>
    static dealii::Vector<real> reconstruct_H1_norm(
        DoFCellAccessorType &                             curr_cell,
        dealii::PolynomialSpace<dim>                      ps,
        dealii::LinearAlgebra::distributed::Vector<real> &solution,
        unsigned int                                      order,
        dealii::hp::MappingCollection<dim> &              mapping_collection,
        dealii::hp::FECollection<dim> &                   fe_collection,
        dealii::hp::QCollection<dim> &                    quadrature_collection,
        dealii::UpdateFlags &                             update_flags);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __RECONSTRUCT_POLY_H__
