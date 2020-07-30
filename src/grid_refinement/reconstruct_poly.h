
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
    // removing the default constructor
    ReconstructPoly() = delete;

    // constructor
    ReconstructPoly(
        const dealii::hp::DoFHandler<dim>&        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>& mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                update_flags);         // update flags for for volume fe

    // reinitialize the vectors
    void reinit(const unsigned int n);

    // sets the norm value
    void set_norm_type(const NormType norm_type);

    // constructs the derivatives along the chords of the cells
    void reconstruct_chord_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,   // solution approximation to be reconstructed
        const unsigned int                                     rel_order); // order of the apporximation

    // construct the largest set of direcitonal derivative (in descending order perpendicular planes)
    void reconstruct_directional_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,   // approximation to be reconstructed
        const unsigned int                                     rel_order); // order of the apporximation

private:
    template <typename DoFCellAccessorType>
    dealii::Vector<real> reconstruct_norm(
        const NormType                                          norm_type,
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution);

    template <typename DoFCellAccessorType>
    dealii::Vector<real> reconstruct_H1_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution);

    template <typename DoFCellAccessorType>
    dealii::Vector<real> reconstruct_L2_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution);

    template <typename DoFCellAccessorType>
    std::vector<DoFCellAccessorType> get_patch_around_dof_cell(
        const DoFCellAccessorType &cell);

    // member attributes
    const dealii::hp::DoFHandler<dim>&         dof_handler;
    const dealii::hp::MappingCollection<dim> & mapping_collection;
    const dealii::hp::FECollection<dim> &      fe_collection;
    const dealii::hp::QCollection<dim> &       quadrature_collection;
    const dealii::UpdateFlags &                update_flags;

    // controls the norm settings
    NormType norm_type;

public:
    // values for return
    std::vector<std::array<real,dim>>                       derivative_value;
    std::vector<std::array<dealii::Tensor<1,dim,real>,dim>> derivative_direction;
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __RECONSTRUCT_POLY_H__
