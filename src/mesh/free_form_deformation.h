#ifndef __SPLINE_CHANNEL__
#define __SPLINE_CHANNEL__

#include "high_order_grid.h"

namespace PHiLiP {

/// Free form deformation class from Sederberg 1986.
template<int dim>
class FreeFormDeformation
{
public:
    /// Constructor for an oblique parallepiped.
    FreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<dealii::Tensor<1,dim,double>,dim> _parallepiped_vectors,
        const std::array<unsigned int,dim> &_ndim_control);

    /// Constructor for a rectangular FFD box.
    FreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<double,dim> &rectangle_lengths,
        const std::array<unsigned int,dim> &_ndim_control);

    std::vector<dealii::Point<dim>> control_pts;

    /// Given a control points' global index return its ijk coordinate.
    /** Opposite of grid_to_global
     */
    std::array<unsigned int,dim> global_to_grid ( const unsigned int global_ictl ) const;

    /// Given a control points' ijk coordinate return its global indexing.
    /** Opposite of global_to_grid
     */
    unsigned int grid_to_global ( const std::array<unsigned int,dim> &ijk_index ) const;
protected:

    /// Returns the local coordinates s-t-u within the FFD box.
    /** s,t,u should be in [0,1] for a point inside the box.
     */
    template<typename real>
    dealii::Point<dim,real> get_local_coordinates (const dealii::Point<dim,real> p) const;

    /// Parallepiped origin.
    const dealii::Point<dim> origin;
    /// Parallepiped vectors.
    /** Not unit vector since the parallepiped length will be determined by those
     *  vectors' magnitude.
     */
    const std::array<dealii::Tensor<1,dim,double>,dim> parallepiped_vectors;

    /// Number of control points in each direction.
    const std::array<unsigned int, dim> ndim_control_pts;
public:
    const unsigned int n_control_pts;
private:

    /// Returns rectangular vector box given the box lengths.
    std::array<dealii::Tensor<1,dim,double>,dim> get_rectangular_parallepiped_vectors (const std::array<double,dim> &rectangle_lengths) const;

    unsigned int compute_total_ctl_pts() const;

    /// Outputs if MPI rank is 0.
    dealii::ConditionalOStream pcout;

    /// Initial message.
    void init_msg() const;
};

} // namespace PHiLiP

#endif
