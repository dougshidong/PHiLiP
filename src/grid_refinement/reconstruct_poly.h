
#ifndef __RECONSTRUCT_POLY_H__
#define __RECONSTRUCT_POLY_H__

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/base/polynomial_space.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

namespace PHiLiP {

namespace GridRefinement {

// norm types availible for reconstruction
enum class NormType{
    H1,
    L2,
    };

// forward declaration of multi-index computation from Dealii
template <int dim>
std::array<unsigned int, dim> compute_index(
    const unsigned int i,
    const unsigned int size);

// funcitons for polynomial reconstruction
template <int dim, int nstate, typename real>
class ReconstructPoly
{
    
public:

    // parser functions that implicitly instantiates cases
    static void reconstruct_chord_derivative(
        const NormType&                                        norm_type,
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
        const dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
        const unsigned int&                                    rel_order,             // order of the reconstruction
        std::vector<dealii::Tensor<1,dim,real>>&               A);                    // holds the reconstructed directional derivative along each centerline chord

    static void reconstruct_directional_derivative(
        const NormType&                                        norm_type,
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
        const dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
        const unsigned int&                                    rel_order,             // order of the reconstruction
        std::vector<dealii::Tensor<1,dim,real>>&               A);                    // (output) holds the largest (scaled) derivative in each direction and then in each orthogonal plane

    template <NormType norm_type = NormType::H1>
    static void reconstruct_chord_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
        const dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
        const unsigned int&                                    rel_order,             // order of the reconstruction
        std::vector<dealii::Tensor<1,dim,real>>&               A);                    // holds the reconstructed directional derivative along each centerline chord

    template <NormType norm_type = NormType::H1>
    static void reconstruct_directional_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
        const dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
        const unsigned int&                                    rel_order,             // order of the reconstruction
        std::vector<dealii::Tensor<1,dim,real>>&               A);                    // (output) holds the largest (scaled) derivative in each direction and then in each orthogonal plane

private:
    template <NormType norm_type = NormType::H1, typename DoFCellAccessorType>
    static dealii::Vector<real> reconstruct_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution,
        const dealii::hp::MappingCollection<dim> &              mapping_collection,
        const dealii::hp::FECollection<dim> &                   fe_collection,
        const dealii::hp::QCollection<dim> &                    quadrature_collection,
        const dealii::UpdateFlags &                             update_flags);

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
