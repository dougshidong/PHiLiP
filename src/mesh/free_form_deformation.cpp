#include <fstream>
#include <boost/math/special_functions/binomial.hpp>

#include <Sacado.hpp>

#include "free_form_deformation.h"
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
FreeFormDeformation<dim>::FreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<dealii::Tensor<1,dim,double>,dim> _parallepiped_vectors,
        const std::array<unsigned int,dim> &_ndim_control_pts)
        : origin(_origin)
        , parallepiped_vectors(_parallepiped_vectors)
        , ndim_control_pts(_ndim_control_pts)
        , n_control_pts(compute_total_ctl_pts())
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{ 
    control_pts.resize(n_control_pts);
    for (unsigned int ictl = 0; ictl < n_control_pts; ++ictl) {

        std::array<unsigned int,dim> ijk = global_to_grid (ictl);

        control_pts[ictl] = origin;
        for (int d=0; d<dim; ++d) {
            control_pts[ictl] += ijk[d] / (ndim_control_pts[d] - 1.0) * parallepiped_vectors[d];
        }

    }
    init_msg();

    pcout << " **************************************** " << std::endl;
}

template<int dim>
FreeFormDeformation<dim>::FreeFormDeformation (
        const dealii::Point<dim> &_origin,
        const std::array<double,dim> &rectangle_lengths,
        const std::array<unsigned int,dim> &_ndim_control_pts)
    : FreeFormDeformation (_origin, get_rectangular_parallepiped_vectors(rectangle_lengths), _ndim_control_pts)
{ }

template<int dim>
dealii::Point<dim,double> FreeFormDeformation<dim>::get_local_coordinates (const dealii::Point<dim,double> p) const
{
    dealii::Point<dim,double> local_coordinates;
    std::array<dealii::Tensor<1,dim,double>,dim> perp_vectors;
    if constexpr (dim == 1) {
        perp_vectors[0] = 1;
    } 
    if constexpr (dim == 2) {

        // X = X_0 + s S + t T
        // s = ( (X - X_0) . T_perp ) / (S . T_perp)
        // t = ( (X - X_0) . S_perp ) / (T . S_perp)
        // In 2D, V_perp = [V_x, V_y]_perp = [V_y, -V_x] = dealii::cross_product_2d(V)
        perp_vectors[0] = dealii::cross_product_2d(parallepiped_vectors[1]);
        perp_vectors[1] = dealii::cross_product_2d(parallepiped_vectors[0]);
    }
    if constexpr (dim == 3) {
        // See Sederberg 1986 for 3D
        perp_vectors[0] = dealii::cross_product_3d(parallepiped_vectors[1], parallepiped_vectors[2]);
        perp_vectors[1] = cross_product_3d(parallepiped_vectors[0], parallepiped_vectors[2]);
        perp_vectors[2] = cross_product_3d(parallepiped_vectors[0], parallepiped_vectors[1]);
    }

    const dealii::Tensor<1,dim,double> dX = (p - origin);
    for (int d=0;d<dim;++d) {
        local_coordinates[d] = (dX * perp_vectors[d]) / (parallepiped_vectors[d] * perp_vectors[d]);
        // const double TOL = 1e-12;
        // if (!(-TOL <= local_coordinates[d] && local_coordinates[d] <= 1.0+TOL)) {
        //     std::cout << " Point: " << p << " is not within the FFD box." << std::endl;
        //     std::cout << " Direction: " << d << " Local (s,t,u) coord: " << local_coordinates[d] << std::endl;
        //     throw(1);
        // }
    }

    return local_coordinates;
}

template<int dim>
std::array<dealii::Tensor<1,dim,double>,dim> FreeFormDeformation<dim>
::get_rectangular_parallepiped_vectors (const std::array<double,dim> &rectangle_lengths) const
{
    std::array<dealii::Tensor<1,dim,double>,dim> parallepiped_vectors;
    for (int d=0; d<dim; ++d) {
        parallepiped_vectors[d][d] = rectangle_lengths[d];
    }
    return parallepiped_vectors;
}

template<int dim>
std::array<unsigned int,dim> FreeFormDeformation<dim>::global_to_grid ( const unsigned int global_ictl ) const
{
    std::array<unsigned int,dim> ijk_index;

    unsigned int remain = global_ictl;
    for (int d=0; d<dim; ++d) {
        ijk_index[d] = remain % ndim_control_pts[d];
        remain /= ndim_control_pts[d];
    }
    assert(remain == 0);

    return ijk_index;
}

template<int dim>
unsigned int FreeFormDeformation<dim>::grid_to_global ( const std::array<unsigned int,dim> &ijk_index ) const
{
    for (int d=0;d<dim;++d) {
        assert(ijk_index[d] < ndim_control_pts[d]);
    }

    unsigned int global_index = 0;
    if constexpr (dim == 1) {
        global_index = ijk_index[0];
    }
    if constexpr (dim == 2) {
        global_index = ijk_index[0];
        global_index += ijk_index[1] * ndim_control_pts[0];
    }
    if constexpr (dim == 3) {
        global_index = ijk_index[0];
        global_index += ijk_index[1] * ndim_control_pts[0];
        global_index += ijk_index[2] * ndim_control_pts[0] * ndim_control_pts[1];
    }
    return global_index;
}


template<int dim>
unsigned int FreeFormDeformation<dim>::compute_total_ctl_pts() const
{
    unsigned int total = 1;
    for (int d=0;d<dim;++d) { total *= ndim_control_pts[d]; }
    return total;
}

template<int dim>
void FreeFormDeformation<dim>::init_msg() const
{
    pcout << " **************************************** " << std::endl;
    pcout << " * Creating Free-Form Deformation (FFD) * " << std::endl;

    pcout << " Parallepiped with corner volume_nodes located at: * " << std::endl;
    for (unsigned int ictl = 0; ictl < n_control_pts; ++ictl) {
        std::array<unsigned int, dim> ijk = global_to_grid(ictl);
        bool print = true;
        for (int d=0; d<dim; ++d) {
            if ( !( ijk[d] == 0 || ijk[d] == (ndim_control_pts[d] - 1) ) ) {
                print = false;
            }
        }
        if (print) pcout << control_pts[ictl] << std::endl;
    }
}

template<int dim>
void FreeFormDeformation<dim>::move_ctl_dx ( const unsigned i, const dealii::Tensor<1,dim,double> dx )
{
    control_pts[i] += dx;
}

template<int dim>
void FreeFormDeformation<dim>::move_ctl_dx ( const std::array<unsigned int,dim> ijk, const dealii::Tensor<1,dim,double> dx)
{
    control_pts[grid_to_global(ijk)] += dx;
}

template<int dim>
dealii::Point<dim,double> FreeFormDeformation<dim>
::new_point_location (const dealii::Point<dim,double> &initial_point) const
{
    return new_point_location (initial_point, this->control_pts);
}


template<int dim>
template<typename real>
dealii::Point<dim,real> FreeFormDeformation<dim>
::new_point_location (
    const dealii::Point<dim,double> &initial_point,
    const std::vector<dealii::Point<dim,real>> &control_pts) const
{
    const dealii::Point<dim,double> s_t_u = get_local_coordinates (initial_point);
    for (int d=0; d<dim; ++d) {
        if (!(0 <= s_t_u[d] && s_t_u[d] <= 1.0)) {
            dealii::Point<dim,real> initial_point_ad;
            for (int d=0; d<dim; ++d) {
                initial_point_ad[d] = initial_point[d];
            }
            return initial_point_ad;
        }
    }
    return evaluate_ffd (s_t_u, control_pts);
}

template<int dim>
template<typename real>
dealii::Tensor<1,dim,real> FreeFormDeformation<dim>
::get_displacement (
    const dealii::Point<dim,double> &initial_point,
    const std::vector<dealii::Point<dim,real>> &control_pts) const
{
    const dealii::Point<dim,real> new_point = new_point_location (initial_point, control_pts);
    const dealii::Tensor<1,dim,real> displacement = new_point - initial_point;
    return displacement;
}


template<int dim>
template<typename real>
std::vector<dealii::Tensor<1,dim,real>> FreeFormDeformation<dim>
::get_displacement (
    const std::vector<dealii::Point<dim,double>> &initial_points,
    const std::vector<dealii::Point<dim,real>> &control_pts) const
{
    std::vector<dealii::Tensor<1,dim,real>> displacements;
    for (unsigned int i=0; i<initial_points.size(); ++i) {
        displacements[i] = get_displacement (initial_points[i], control_pts);
    }
    return displacements;
}

template<int dim>
dealii::Point<dim,double> FreeFormDeformation<dim>
::dXdXp (const dealii::Point<dim,double> &initial_point, const unsigned int ctl_index, const unsigned int ctl_axis) const
{
    assert(ctl_axis < dim);
    assert(ctl_index < n_control_pts);
    using FadType = Sacado::Fad::DFad<double>;
    std::vector<dealii::Point<dim,FadType>> control_pts_ad(control_pts.size());
    for (unsigned int i=0; i<n_control_pts; ++i) {
        control_pts_ad[i] = control_pts[i];
    }
    control_pts_ad[ctl_index][ctl_axis].diff(0,1);

    dealii::Point<dim, FadType> new_point_ad = new_point_location(initial_point, control_pts_ad);

    dealii::Point<dim,double> dXdXp;
    for (int d=0; d<dim; ++d) {
        dXdXp[d] = new_point_ad[d].dx(0);
    }
    return dXdXp;
}

template<int dim>
template<typename real>
dealii::Point<dim,real> FreeFormDeformation<dim>
::evaluate_ffd (
    const dealii::Point<dim,double> &s_t_u_point,
    const std::vector<dealii::Point<dim,real>> &control_pts) const
{
    dealii::Point<dim,real> ffd_location;

    std::array<std::vector<real>,dim> ijk_coefficients;
    
    for (int d=0; d<dim; ++d) {
        ffd_location[d] = 0.0;

        ijk_coefficients[d].resize(ndim_control_pts[d]);

        const unsigned n_intervals = ndim_control_pts[d] - 1;

        for (unsigned int i = 0; i < ndim_control_pts[d]; ++i) {
            double bin_coeff = boost::math::binomial_coefficient<double>(n_intervals, i);
            const unsigned int power = n_intervals - i;
            ijk_coefficients[d][i] = bin_coeff * std::pow(1.0 - s_t_u_point[d], power) * std::pow(s_t_u_point[d], i);
        }
    }

    for (unsigned int ictl = 0; ictl < n_control_pts; ++ictl) {
        std::array<unsigned int, dim> ijk = global_to_grid(ictl);

        real coeff = 1.0;
        for (int d=0; d<dim; ++d) {
            coeff *= ijk_coefficients[d][ijk[d]];
        }
        for (int d=0; d<dim; ++d) {
            ffd_location[d] += coeff * control_pts[ictl][d];
        }
    }

    return ffd_location;
}

template<int dim>
void FreeFormDeformation<dim>
::get_design_variables(
    const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
    dealii::LinearAlgebra::distributed::Vector<double> &vector_to_copy_into) const
{
    AssertDimension(ffd_design_variables_indices_dim.size(), vector_to_copy_into.size())
    auto partitioner = vector_to_copy_into.get_partitioner();

    for (unsigned int i_dvar = 0; i_dvar < ffd_design_variables_indices_dim.size(); ++i_dvar) {
        if (partitioner->in_local_range(i_dvar)) {
            const unsigned int i_ctl = ffd_design_variables_indices_dim[i_dvar].first;
            const unsigned int d_ctl = ffd_design_variables_indices_dim[i_dvar].second;
            vector_to_copy_into[i_dvar] = this->control_pts[i_ctl][d_ctl];
        }
    }
    vector_to_copy_into.update_ghost_values();
}

template<int dim>
void FreeFormDeformation<dim>
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
}

template<int dim>
void FreeFormDeformation<dim>
::deform_mesh (HighOrderGrid<dim,double> &high_order_grid) const
{
    dealii::LinearAlgebra::distributed::Vector<double>  surface_node_displacements = get_surface_displacement (high_order_grid);

    MeshMover::LinearElasticity<dim, double>
        meshmover( 
          *(high_order_grid.triangulation),
          high_order_grid.initial_mapping_fe_field,
          high_order_grid.dof_handler_grid,
          high_order_grid.surface_to_volume_indices,
          surface_node_displacements);
    dealii::LinearAlgebra::distributed::Vector<double> volume_displacements = meshmover.get_volume_displacements();
    high_order_grid.volume_nodes = high_order_grid.initial_volume_nodes;
    high_order_grid.volume_nodes += volume_displacements;
    high_order_grid.volume_nodes.update_ghost_values();
}

template<int dim>
dealii::LinearAlgebra::distributed::Vector<double> 
FreeFormDeformation<dim>
::get_surface_displacement (const HighOrderGrid<dim,double> &high_order_grid) const
{
    dealii::LinearAlgebra::distributed::Vector<double> surface_node_displacements(high_order_grid.surface_nodes);

    auto index = high_order_grid.surface_to_volume_indices.begin();
    auto node = high_order_grid.initial_surface_nodes.begin();
    auto new_node = surface_node_displacements.begin();
    for (; index != high_order_grid.surface_to_volume_indices.end(); ++index, ++node, ++new_node) {
        const dealii::types::global_dof_index global_idof_index = *index;
        const std::pair<unsigned int, unsigned int> ipoint_component = high_order_grid.global_index_to_point_and_axis.at(global_idof_index);
        const unsigned int ipoint = ipoint_component.first;
        const unsigned int component = ipoint_component.second;
        dealii::Point<dim> old_point;
        for (int d=0;d<dim;d++) {
            old_point[d] = high_order_grid.initial_locally_relevant_surface_points[ipoint][d];
        }
        const dealii::Point<dim> new_point = new_point_location(old_point);
        *new_node = new_point[component];
    }
    surface_node_displacements.update_ghost_values();
    surface_node_displacements -= high_order_grid.initial_surface_nodes;
    surface_node_displacements.update_ghost_values();

    return surface_node_displacements;
}

template<int dim>
std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
FreeFormDeformation<dim>
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
        const dealii::Point<dim> old_ffd_point = control_pts[ictl];

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] += eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
        }
        const auto surface_node_displacements_p = get_surface_displacement (high_order_grid);

        // Reset FFD
        control_pts[ictl] = old_ffd_point;

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] -= eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
        }
        const auto surface_node_displacements_n = get_surface_displacement (high_order_grid);

        // Reset FFD
        control_pts[ictl] = old_ffd_point;

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
std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
FreeFormDeformation<dim>
::get_dXvsdXp (
    const HighOrderGrid<dim,double> &high_order_grid,
    const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim
    ) const
{
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvsdXp_vector;

    const dealii::IndexSet &nodes_locally_owned = high_order_grid.volume_nodes.get_partitioner()->locally_owned_range();
    for (auto const &ffd_pair: ffd_design_variables_indices_dim) {

        const unsigned int ctl_index = ffd_pair.first;
        const unsigned int ctl_axis  = ffd_pair.second;

        dealii::LinearAlgebra::distributed::Vector<double> derivative_surface_nodes_ffd_ctl;
        derivative_surface_nodes_ffd_ctl.reinit(high_order_grid.volume_nodes);
        unsigned int ipoint = 0;
        for (auto const& surface_point: high_order_grid.initial_locally_relevant_surface_points) {

            dealii::Point<dim,double> dxsdxp = dXdXp (surface_point, ctl_index, ctl_axis);

            for (int d=0; d<dim; ++d) { 
                if ((unsigned int)d!=ctl_axis) {
                    assert(dxsdxp[d] == 0.0);
                }
                const dealii::types::global_dof_index vol_index = high_order_grid.point_and_axis_to_global_index.at(std::make_pair(ipoint,(unsigned int)d));
                if (nodes_locally_owned.is_element(vol_index)) {
                    derivative_surface_nodes_ffd_ctl[vol_index] = dxsdxp[d];
                }
            }

            ipoint++;
        }
        derivative_surface_nodes_ffd_ctl.update_ghost_values();

        dXvsdXp_vector.push_back(derivative_surface_nodes_ffd_ctl);
    }
    return dXvsdXp_vector;
}

template<int dim>
void
FreeFormDeformation<dim>
::get_dXvsdXp (
    const HighOrderGrid<dim,double> &high_order_grid,
    const std::vector< std::pair< unsigned int, unsigned int > > &ffd_design_variables_indices_dim,
    dealii::TrilinosWrappers::SparseMatrix &dXvsdXp
    ) const
{
    const unsigned int n_rows = high_order_grid.dof_handler_grid.n_dofs();
    const unsigned int n_cols = ffd_design_variables_indices_dim.size();
    const dealii::IndexSet &row_part = high_order_grid.dof_handler_grid.locally_owned_dofs();
    const dealii::IndexSet col_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_cols);

    dealii::DynamicSparsityPattern full_dsp(n_rows, n_cols, row_part);
    for (const auto &i_row: row_part) {
        for (unsigned int i_col = 0; i_col < n_cols; ++i_col) {
            full_dsp.add(i_row, i_col);
        }
    }
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(high_order_grid.dof_handler_grid, locally_relevant_dofs);
    dealii::SparsityTools::distribute_sparsity_pattern(full_dsp, row_part, MPI_COMM_WORLD, locally_relevant_dofs);

    dealii::SparsityPattern full_sp;
    full_sp.copy_from(full_dsp);

    dXvsdXp.reinit(row_part, col_part, full_sp, MPI_COMM_WORLD);


    const dealii::IndexSet &nodes_locally_owned = high_order_grid.volume_nodes.get_partitioner()->locally_owned_range();
    for (unsigned int i_col = 0; i_col < ffd_design_variables_indices_dim.size(); ++i_col) {

        const auto ffd_pair = ffd_design_variables_indices_dim[i_col];
        const unsigned int ctl_index = ffd_pair.first;
        const unsigned int ctl_axis  = ffd_pair.second;

        unsigned int ipoint = 0;
        for (auto const& surface_point: high_order_grid.initial_locally_relevant_surface_points) {

            dealii::Point<dim,double> dxsdxp = dXdXp (surface_point, ctl_index, ctl_axis);

            for (int d=0; d<dim; ++d) { 
                const dealii::types::global_dof_index vol_index = high_order_grid.point_and_axis_to_global_index.at(std::make_pair(ipoint,(unsigned int)d));
                if (nodes_locally_owned.is_element(vol_index)) {
                    dXvsdXp.set(vol_index,i_col, dxsdxp[d]);
                }
                if ((unsigned int)d!=ctl_axis) {
                    assert(dxsdxp[d] == 0.0);
                }
            }

            ipoint++;
        }
    }
    dXvsdXp.compress(dealii::VectorOperation::insert);
}

template<int dim>
void
FreeFormDeformation<dim>
::get_dXvdXp (
    const HighOrderGrid<dim,double> &high_order_grid,
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
    dealii::TrilinosWrappers::SparseMatrix &dXvdXp
    ) const
{
    // const unsigned int n_design_var = ffd_design_variables_indices_dim.size();

    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvsdXp_vector = get_dXvsdXp(high_order_grid, ffd_design_variables_indices_dim);

    dealii::LinearAlgebra::distributed::Vector<double> surface_node_displacements(high_order_grid.surface_nodes);
    MeshMover::LinearElasticity<dim, double>
        meshmover( 
          *(high_order_grid.triangulation),
          high_order_grid.initial_mapping_fe_field,
          high_order_grid.dof_handler_grid,
          high_order_grid.surface_to_volume_indices,
          surface_node_displacements);
    //meshmover.evaluate_dXvdXs();
    meshmover.apply_dXvdXvs(dXvsdXp_vector, dXvdXp);
}

template<int dim>
void
FreeFormDeformation<dim>
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
        const dealii::Point<dim> old_ffd_point = control_pts[ictl];

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] += eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
            deform_mesh(high_order_grid);
        }

        auto nodes_p = high_order_grid.volume_nodes;

        // Reset FFD
        control_pts[ictl] = old_ffd_point;
        high_order_grid.volume_nodes = old_volume_nodes;

        // Perturb
        {
            dealii::Point<dim> new_ffd_point = old_ffd_point;
            new_ffd_point[d_ffd] -= eps;
            move_ctl_dx ( ictl, new_ffd_point - old_ffd_point);
            deform_mesh(high_order_grid);
        }

        auto nodes_m = high_order_grid.volume_nodes;

        // Reset FFD
        control_pts[ictl] = old_ffd_point;
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

template<int dim>
void FreeFormDeformation<dim>
::output_ffd_vtu(const unsigned int cycle) const
{
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0) return;
    // next create the cells
    std::vector<dealii::CellData<dim>> cells;
    unsigned int n_cells = 1;
    for (int d = 0; d<dim; ++d) {
        n_cells *= ndim_control_pts[d]-1;
    }
    cells.resize(n_cells);

    // From here, the code is copied from dealii::GridGenerator::subdivided_parallelpiped()
    std::array<unsigned int, dim> repetitions;
    for (int d = 0; d < dim; ++d) {
        repetitions[d] = ndim_control_pts[d]-1;
    }
    switch (dim) {
        case 1:
        {
            for (unsigned int x = 0; x < repetitions[0]; ++x) {
                cells[x].vertices[0] = x;
                cells[x].vertices[1] = x + 1;
                cells[x].material_id = 0;
            }
            break;
        }
 
        case 2:
        {
            for (unsigned int y = 0; y < repetitions[1]; ++y) {
                for (unsigned int x = 0; x < repetitions[0]; ++x) {
                    const unsigned int c = x + y * repetitions[0];
                    cells[c].vertices[0] = y * (repetitions[0] + 1) + x;
                    cells[c].vertices[1] = y * (repetitions[0] + 1) + x + 1;
                    cells[c].vertices[2] = (y + 1) * (repetitions[0] + 1) + x;
                    cells[c].vertices[3] = (y + 1) * (repetitions[0] + 1) + x + 1;
                    cells[c].material_id = 0;
                }
            }
            break;
        }
 
        case 3:
        {
            const unsigned int n_x = (repetitions[0] + 1);
            const unsigned int n_xy = (repetitions[0] + 1) * (repetitions[1] + 1);
 
            for (unsigned int z = 0; z < repetitions[2]; ++z) {
                for (unsigned int y = 0; y < repetitions[1]; ++y) {
                    for (unsigned int x = 0; x < repetitions[0]; ++x) {
                        const unsigned int c = x + y * repetitions[0] + z * repetitions[0] * repetitions[1];
                        cells[c].vertices[0] = z * n_xy + y * n_x + x;
                        cells[c].vertices[1] = z * n_xy + y * n_x + x + 1;
                        cells[c].vertices[2] = z * n_xy + (y + 1) * n_x + x;
                        cells[c].vertices[3] = z * n_xy + (y + 1) * n_x + x + 1;
                        cells[c].vertices[4] = (z + 1) * n_xy + y * n_x + x;
                        cells[c].vertices[5] = (z + 1) * n_xy + y * n_x + x + 1;
                        cells[c].vertices[6] = (z + 1) * n_xy + (y + 1) * n_x + x;
                        cells[c].vertices[7] = (z + 1) * n_xy + (y + 1) * n_x + x + 1;
                        cells[c].material_id = 0;
                    }
                }
            }
            break;
        }
    } // switch(dim)
 
    dealii::GridReordering<dim>::reorder_cells(cells, true);
    dealii::Triangulation<dim,dim> tria;
    tria.create_triangulation(control_pts, cells, dealii::SubCellData());
    std::string nffd_string[3];
    for (int d=0; d<dim; ++d) {
        nffd_string[d] = dealii::Utilities::int_to_string(ndim_control_pts[d], 3);
    }
    std::string filename = "FFD-" + dealii::Utilities::int_to_string(dim, 1) +"D_";
    for (int d=0; d<dim; ++d) {
        filename += dealii::Utilities::int_to_string(ndim_control_pts[d], 3);
        if (d<dim-1) filename += "X";
    }
    filename += "-"+dealii::Utilities::int_to_string(cycle, 4) + ".vtu";
    pcout << "Outputting FFD grid: " << filename << " ... " << std::endl;

    std::ofstream output(filename);

    dealii::GridOut grid_out;
    grid_out.write_vtu (tria, output);
 
}


template class FreeFormDeformation<PHILIP_DIM>;

// template dealii::Point<dim,double> FreeFormDeformation<dim>
// ::evaluate_ffd (const dealii::Point<PHILIP_DIM,double> &, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;
// 
// template dealii::Point<PHILIP_DIM, double> FreeFormDeformation<PHILIP_DIM>
// ::new_point_location(const dealii::Point<PHILIP_DIM, double>&, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;
// 
// template dealii::Tensor<1, PHILIP_DIM, double> FreeFormDeformation<PHILIP_DIM>
// ::get_displacement (const dealii::Point<PHILIP_DIM,double> &, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;
// 
// template std::vector<dealii::Tensor<1, PHILIP_DIM, double>> FreeFormDeformation<PHILIP_DIM>
// ::get_displacement (const std::vector<dealii::Point<PHILIP_DIM,double>> &, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;

} // namespace PHiLiP
