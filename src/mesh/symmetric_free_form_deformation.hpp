#ifndef __SYMMETRIC_FREE_FORM_DEFORMATION__
#define __SYMMETRIC_FREE_FORM_DEFORMATION__

#include "free_form_deformation.h"
#include "high_order_grid.h"

namespace PHiLiP {

/// Free form deformation class from Sederberg 1986.
template<int dim>
class SymmetryFreeFormDeformation : public FreeFormDeformation<dim>
{
public:
    /// Constructor for an oblique parallepiped.
    SymmetryFreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<dealii::Tensor<1,dim,double>,dim> _parallepiped_vectors,
        const std::array<unsigned int,dim> &_ndim_control);

    /// Constructor for a rectangular FFD box.
    SymmetryFreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<double,dim> &rectangle_lengths,
        const std::array<unsigned int,dim> &_ndim_control);

    SymmetryFreeFormDeformation (const FreeFormDeformation<dim> &ffd);

    /// Given an initial point in the undeformed initial parallepiped and the index a control point,
    /// return the derivative dXdXp of the new point location point_i with respect to that control_point_j.
    dealii::Point<dim,double> dXdXp (const dealii::Point<dim,double> &initial_point, const unsigned int ctl_index, const unsigned int ctl_axis) const override;

    /** For the given list of FFD indices and direction, return the finite-differenced
     *  derivatives of the HighOrderGrid's initial surface points with respect to the FFD.
     *  Note that the result is returned as a vector of vector because it will likely be used with a MeshMover's apply_dXvdXvs() function.
     */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
    get_dXvsdXp_FD (const HighOrderGrid<dim,double> &high_order_grid,
                const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
                const double eps
               ) override;

    /** For the given list of FFD indices and direction, return the analytical
     *  derivatives of the HighOrderGrid's initial volume points with respect to the FFD.
     */
    void
    get_dXvdXp_FD (HighOrderGrid<dim,double> &high_order_grid,
                const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
                dealii::TrilinosWrappers::SparseMatrix &dXvdXp_FD,
                const double eps
                ) override;

    /// Move control point with global index i.
    void move_ctl_dx ( const unsigned i, const dealii::Tensor<1,dim,double> ) override;

    /// Move control point with grid index (i,j,k).
    void move_ctl_dx ( const std::array<unsigned int,dim> ijk, const dealii::Tensor<1,dim,double> ) override;

    /// Copies the desired control points from vector_to_copy_from into FreeFormDeformation object 
    void set_design_variables(
        const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
        dealii::LinearAlgebra::distributed::Vector<double> &vector_to_copy_from) override;

protected:

    template<typename real>
    std::vector<dealii::Point<dim,real>> symmetrize(bool upper, const std::vector<dealii::Point<dim,real>>& control_pts) const;

    unsigned int symmetric_y_id(const unsigned int ictl) const;

};

} // namespace PHiLiP

#endif
