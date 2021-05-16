#ifndef __FREE_FORM_DEFORMATION__
#define __FREE_FORM_DEFORMATION__

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

    /// Given an initial point in the undeformed initial parallepiped, return the 
    /// position of the new point location.
    template<typename real>
    dealii::Point<dim,real> new_point_location
        (const dealii::Point<dim,double> &initial_point,
         const std::vector<dealii::Point<dim,real>> &control_pts) const;

    /// Given an initial point in the undeformed initial parallepiped, return the 
    /// position of the new point location using the current control point locations.
    dealii::Point<dim,double> new_point_location (const dealii::Point<dim,double> &initial_point) const;

    /// Using the initial surface nodes from the given HighOrderGrid, return the surface displacements based
    /// on the free-form deformation displacements.
    dealii::LinearAlgebra::distributed::Vector<double> 
    get_surface_displacement (const HighOrderGrid<dim,double> &high_order_grid) const;

    /// Deform HighOrderGrid using its initial volume_nodes to retrieve the deformed set of volume_nodes.
    void deform_mesh (HighOrderGrid<dim,double> &high_order_grid) const;

    /// Given an initial point in the undeformed initial parallepiped and the index a control point,
    /// return the derivative dXdXp of the new point location point_i with respect to that control_point_j.
    dealii::Point<dim,double> dXdXp (const dealii::Point<dim,double> &initial_point, const unsigned int ctl_index, const unsigned int ctl_axis) const;

    /** For the given list of FFD indices and direction, return the analytical
     *  derivatives of the HighOrderGrid's initial surface points with respect to the FFD.
     *  The result is written into the given dXvsdXp SparseMatrix.
     */
    void get_dXvsdXp (
        const HighOrderGrid<dim,double> &high_order_grid,
        const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
        dealii::TrilinosWrappers::SparseMatrix &dXvsdXp
        ) const;

    /** For the given list of FFD indices and direction, return the analytical
     *  derivatives of the HighOrderGrid's initial surface points with respect to the FFD.
     *  Note that the result is returned as a vector of vector because it will likely be used with a MeshMover's apply_dXvdXvs() function.
     */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
    get_dXvsdXp (const HighOrderGrid<dim,double> &high_order_grid,
                const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim
               ) const;

    /** For the given list of FFD indices and direction, return the finite-differenced
     *  derivatives of the HighOrderGrid's initial surface points with respect to the FFD.
     *  Note that the result is returned as a vector of vector because it will likely be used with a MeshMover's apply_dXvdXvs() function.
     */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
    get_dXvsdXp_FD (const HighOrderGrid<dim,double> &high_order_grid,
                const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
                const double eps
               );

    /** For the given list of FFD indices and direction, return the finite-differenced
     *  derivatives of the HighOrderGrid's initial volume points with respect to the FFD.
     */
    void
    get_dXvdXp (const HighOrderGrid<dim,double> &high_order_grid,
                const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
                dealii::TrilinosWrappers::SparseMatrix &dXvdXp
                ) const;
    /** For the given list of FFD indices and direction, return the analytical
     *  derivatives of the HighOrderGrid's initial volume points with respect to the FFD.
     */
    void
    get_dXvdXp_FD (HighOrderGrid<dim,double> &high_order_grid,
                const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
                dealii::TrilinosWrappers::SparseMatrix &dXvdXp_FD,
                const double eps
                );

    /// Given an initial point in the undeformed initial parallepiped, return the 
    /// its displacement due to the free-form deformation.
    template<typename real>
    dealii::Tensor<1,dim,real> get_displacement (
        const dealii::Point<dim,double> &initial_point,
        const std::vector<dealii::Point<dim,real>> &control_pts) const;

    /// Given a vector of initial points in the undeformed initial parallepiped, return the 
    /// its corresponding displacements due to the free-form deformation.
    template<typename real>
    std::vector<dealii::Tensor<1,dim,real>> get_displacement (
        const std::vector<dealii::Point<dim,double>> &initial_point,
        const std::vector<dealii::Point<dim,real>> &control_pts) const;

    /// Given the s,t,u reference location within the FFD box, return its position in the 
    /// actual domain.
    template<typename real>
    dealii::Point<dim,real> evaluate_ffd (
        const dealii::Point<dim,double> &s_t_u_point,
        const std::vector<dealii::Point<dim,real>> &control_pts) const;

    /// Control points of the FFD box used to deform the geometry.
    std::vector<dealii::Point<dim>> control_pts;

    /// Given a control points' global index return its ijk coordinate.
    /** Opposite of grid_to_global
     */
    std::array<unsigned int,dim> global_to_grid ( const unsigned int global_ictl ) const;

    /// Given a control points' ijk coordinate return its global indexing.
    /** Opposite of global_to_grid
     */
    unsigned int grid_to_global ( const std::array<unsigned int,dim> &ijk_index ) const;

    /// Move control point with global index i.
    void move_ctl_dx ( const unsigned i, const dealii::Tensor<1,dim,double> );

    /// Move control point with grid index (i,j,k).
    void move_ctl_dx ( const std::array<unsigned int,dim> ijk, const dealii::Tensor<1,dim,double> );

    /// Copies the desired control points from FreeFormDeformation object into vector_to_copy_into.
    void get_design_variables(
        const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
        dealii::LinearAlgebra::distributed::Vector<double> &vector_to_copy_into) const;
    /// Copies the desired control points from vector_to_copy_from into FreeFormDeformation object 
    void set_design_variables(
        const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
        dealii::LinearAlgebra::distributed::Vector<double> &vector_to_copy_from);

    /// Output a .vtu file of the FFD box to visualize.
    void output_ffd_vtu(const unsigned int cycle) const;

protected:

    /// Returns the local coordinates s-t-u within the FFD box.
    /** s,t,u should be in [0,1] for a point inside the box.
     */
    dealii::Point<dim,double> get_local_coordinates (const dealii::Point<dim,double> p) const;

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
    /// Total number of control points.
    const unsigned int n_control_pts;
private:

    /// Returns rectangular vector box given the box lengths.
    std::array<dealii::Tensor<1,dim,double>,dim> get_rectangular_parallepiped_vectors (const std::array<double,dim> &rectangle_lengths) const;

    /// Used by constructor to evaluate total number of control points.
    unsigned int compute_total_ctl_pts() const;

    /// Outputs if MPI rank is 0.
    dealii::ConditionalOStream pcout;

    /// Initial message.
    void init_msg() const;
};

} // namespace PHiLiP

#endif
