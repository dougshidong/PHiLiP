#include <fstream>
#include <boost/math/special_functions/binomial.hpp>

#include <Sacado.hpp>

#include "symmetric_free_form_deformation.hpp"
#include "meshmover_linear_elasticity.hpp"

#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_reordering.h>

// For FD
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
unsigned int SymmetryFreeFormDeformation<dim>::symmetric_y_id(const unsigned int ictl) const
{
    const std::array<unsigned int, dim> ictl_grid = this->global_to_grid (ictl);
    std::array<unsigned int,dim> ictl_grid_symmetry = ictl_grid;
    ictl_grid_symmetry[1] = this->ndim_control_pts[1] - 1 - ictl_grid[1];
    return this->grid_to_global(ictl_grid_symmetry);
}


template<int dim>
template<typename real>
std::vector<dealii::Point<dim,real>>
SymmetryFreeFormDeformation<dim>::symmetrize(const bool upper, const std::vector<dealii::Point<dim,real>>& control_pts) const
{
    std::vector<dealii::Point<dim,real>> pts = control_pts;

    for (unsigned int ictl=0; ictl<this->n_control_pts; ++ictl) {
        const unsigned int ictl_symmetry = symmetric_y_id(ictl);
        if (upper) {
            if (this->global_to_grid(ictl)[1] < this->global_to_grid(ictl_symmetry)[1]) continue;
        } else {
            if (this->global_to_grid(ictl)[1] > this->global_to_grid(ictl_symmetry)[1]) continue;
        }

        pts[ictl_symmetry] = control_pts[ictl];

        double center = this->origin[1] + this->parallepiped_vectors[1][1] * 0.5;
        real distance_from_centerline = control_pts[ictl][1] - center;

        pts[ictl_symmetry][1] = center - distance_from_centerline;
    }
    return pts;
}

template<int dim>
SymmetryFreeFormDeformation<dim>::SymmetryFreeFormDeformation (const FreeFormDeformation<dim> &ffd)
        : FreeFormDeformation<dim>(ffd)
{ 
}

template<int dim>
SymmetryFreeFormDeformation<dim>::SymmetryFreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<dealii::Tensor<1,dim,double>,dim> _parallepiped_vectors,
        const std::array<unsigned int,dim> &_ndim_control_pts)
        : FreeFormDeformation<dim>(_origin, _parallepiped_vectors, _ndim_control_pts)
{ 
}

template<int dim>
SymmetryFreeFormDeformation<dim>::SymmetryFreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<double,dim> &rectangle_lengths,
        const std::array<unsigned int,dim> &_ndim_control_pts)
    : SymmetryFreeFormDeformation (_origin, this->get_rectangular_parallepiped_vectors(rectangle_lengths), _ndim_control_pts)
{ }

template<int dim>
void SymmetryFreeFormDeformation<dim>::move_ctl_dx ( const unsigned i, const dealii::Tensor<1,dim,double> dx )
{
    this->control_pts[i] += dx;
    this->control_pts = symmetrize(true, this->control_pts);
}

template<int dim>
void SymmetryFreeFormDeformation<dim>::move_ctl_dx ( const std::array<unsigned int,dim> ijk, const dealii::Tensor<1,dim,double> dx)
{
    this->control_pts[this->grid_to_global(ijk)] += dx;
    this->control_pts = symmetrize(true, this->control_pts);
}

template<int dim>
dealii::Point<dim,double> SymmetryFreeFormDeformation<dim>
::dXdXp (const dealii::Point<dim,double> &initial_point, const unsigned int ctl_index, const unsigned int ctl_axis) const
{
    assert(ctl_axis < dim);
    assert(ctl_index < this->n_control_pts);
    using FadType = Sacado::Fad::DFad<double>;
    std::vector<dealii::Point<dim,FadType>> control_pts_ad(this->control_pts.size());
    for (unsigned int i=0; i<this->n_control_pts; ++i) {
        control_pts_ad[i] = this->control_pts[i];
    }
    control_pts_ad[ctl_index][ctl_axis].diff(0,1);

    control_pts_ad = symmetrize(true, control_pts_ad);

    dealii::Point<dim, FadType> new_point_ad = this->new_point_location(initial_point, control_pts_ad);

    dealii::Point<dim,double> dXdXp;
    for (int d=0; d<dim; ++d) {
        dXdXp[d] = new_point_ad[d].dx(0);
    }
    return dXdXp;
}


template<int dim>
void SymmetryFreeFormDeformation<dim>
::set_design_variables(
    const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
    dealii::LinearAlgebra::distributed::Vector<double> &vector_to_copy_from)
{
    vector_to_copy_from.update_ghost_values();
    AssertDimension(ffd_design_variables_indices_dim.size(), vector_to_copy_from.size())
    auto partitioner = vector_to_copy_from.get_partitioner();
    for (unsigned int i_dvar = 0; i_dvar < ffd_design_variables_indices_dim.size(); ++i_dvar) {

        assert( partitioner->in_local_range(i_dvar) || partitioner->is_ghost_entry(i_dvar) );

        const unsigned int i_ctl = ffd_design_variables_indices_dim[i_dvar].first;
        const unsigned int d_ctl = ffd_design_variables_indices_dim[i_dvar].second;
        this->control_pts[i_ctl][d_ctl] = vector_to_copy_from[i_dvar];
    }
    this->control_pts = symmetrize(true, this->control_pts);
}


template<int dim>
std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
SymmetryFreeFormDeformation<dim>
::get_dXvsdXp_FD (
    const HighOrderGrid<dim,double> &high_order_grid,
    const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
    const double eps
    )
{
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvsdXp_vector_FD;

    for (auto const &ffd_pair: ffd_design_variables_indices_dim) {

        const unsigned int ictl = ffd_pair.first;
        const unsigned int d_ffd  = ffd_pair.second;

        // Save
        const dealii::Point<dim> old_ffd_point = this->control_pts[ictl];

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] += eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
        }
        const auto surface_node_displacements_p = this->get_surface_displacement (high_order_grid);

        // Reset FFD
        this->control_pts[ictl] = old_ffd_point;
        this->control_pts = symmetrize(true, this->control_pts);

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] -= eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
        }
        const auto surface_node_displacements_n = this->get_surface_displacement (high_order_grid);

        // Reset FFD
        this->control_pts[ictl] = old_ffd_point;
        this->control_pts = symmetrize(true, this->control_pts);

        auto diff = surface_node_displacements_p;
        diff -= surface_node_displacements_n;
        diff /= 2.0*eps;

        // Put into volume-sized vector
        dealii::LinearAlgebra::distributed::Vector<double> derivative_surface_nodes_ffd_ctl;
        derivative_surface_nodes_ffd_ctl.reinit(high_order_grid.volume_nodes);

        high_order_grid.map_nodes_surf_to_vol.vmult(derivative_surface_nodes_ffd_ctl,diff);

        // auto surf_index = high_order_grid.surface_to_volume_indices.begin();
        // auto surf_value = diff.begin();

        // for (; surf_index != high_order_grid.surface_to_volume_indices.end(); ++surf_index, ++surf_value) {
        //     derivative_surface_nodes_ffd_ctl[*surf_index] = *surf_value;
        // }

        // derivative_surface_nodes_ffd_ctl.update_ghost_values();

        dXvsdXp_vector_FD.push_back(derivative_surface_nodes_ffd_ctl);
    }
    return dXvsdXp_vector_FD;
}

template<int dim>
void
SymmetryFreeFormDeformation<dim>
::get_dXvdXp_FD (
    HighOrderGrid<dim,double> &high_order_grid,
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
    dealii::TrilinosWrappers::SparseMatrix &dXvdXp_FD,
    const double eps
    )
{
    const unsigned int n_rows = high_order_grid.volume_nodes.size();
    const unsigned int n_cols = ffd_design_variables_indices_dim.size();

    // Row partitioning
    const dealii::IndexSet &row_part = high_order_grid.dof_handler_grid.locally_owned_dofs();

    //const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    //const std::vector<dealii::IndexSet> col_parts = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_cols);
    //const dealii::IndexSet &col_part = col_parts[this_mpi_process];
    const dealii::IndexSet col_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_cols);

    // Sparsity pattern
    dealii::DynamicSparsityPattern full_dsp(n_rows, n_cols, row_part);
    for (const auto &i_row: row_part) {
        for (unsigned int i_col = 0; i_col < n_cols; ++i_col) {
            full_dsp.add(i_row, i_col);
        }
    }
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(high_order_grid.dof_handler_grid, locally_relevant_dofs);
    dealii::SparsityTools::distribute_sparsity_pattern(full_dsp, high_order_grid.dof_handler_grid.locally_owned_dofs(), MPI_COMM_WORLD, locally_relevant_dofs);
    dealii::SparsityPattern full_sp;
    full_sp.copy_from(full_dsp);

    // Allocate matrix
    dXvdXp_FD.reinit(row_part, col_part, full_sp, MPI_COMM_WORLD);

    // Get finite-differenced dXvdXp_FD
    auto old_volume_nodes = high_order_grid.volume_nodes;
    for (unsigned int i_design = 0; i_design < ffd_design_variables_indices_dim.size(); ++i_design) {
        const unsigned int ictl = ffd_design_variables_indices_dim[i_design].first;
        const unsigned int d_ffd = ffd_design_variables_indices_dim[i_design].second;


        // Save
        const dealii::Point<dim> old_ffd_point = this->control_pts[ictl];

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] += eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
            this->deform_mesh(high_order_grid);
        }

        auto nodes_p = high_order_grid.volume_nodes;

        // Reset FFD
        this->control_pts[ictl] = old_ffd_point;
        this->control_pts = symmetrize(true, this->control_pts);

        high_order_grid.volume_nodes = old_volume_nodes;

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] -= eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
            this->deform_mesh(high_order_grid);
        }

        auto nodes_m = high_order_grid.volume_nodes;

        // Reset FFD
        this->control_pts[ictl] = old_ffd_point;
        this->control_pts = symmetrize(true, this->control_pts);
        high_order_grid.volume_nodes = old_volume_nodes;

        auto dXvdXp_i = nodes_p;
        dXvdXp_i -= nodes_m;
        dXvdXp_i /= (2*eps);

        const auto &locally_owned = dXvdXp_i.get_partitioner()->locally_owned_range();
        for (const auto &index: locally_owned) {
            dXvdXp_FD.set(index,i_design, dXvdXp_i[index]);
        }
    }
    dXvdXp_FD.compress(dealii::VectorOperation::insert);
}



template class SymmetryFreeFormDeformation<PHILIP_DIM>;

} // namespace PHiLiP

