#include <boost/math/special_functions/binomial.hpp>

#include <Sacado.hpp>

#include "free_form_deformation.h"
#include "meshmover_linear_elasticity.hpp"


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

    unsigned int remainder = global_ictl;
    for (int d=0; d<dim; ++d) {
        ijk_index[d] = remainder % ndim_control_pts[d];
        remainder /= ndim_control_pts[d];
    }
    assert(remainder == 0);

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

    pcout << " Parallepiped with corner nodes located at: * " << std::endl;
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
::dXsdXp (const dealii::Point<dim,double> &initial_point, const unsigned int ctl_index, const unsigned int ctl_axis) const
{
    assert(ctl_axis < dim);
    assert(ctl_index < n_control_pts);
    using ADtype = Sacado::Fad::DFad<double>;
    std::vector<dealii::Point<dim,ADtype>> control_pts_ad(control_pts.size());
    for (unsigned int i=0; i<n_control_pts; ++i) {
        control_pts_ad[i] = control_pts[i];
    }
    control_pts_ad[ctl_index][ctl_axis].diff(0,1);

    dealii::Point<dim, ADtype> new_point_ad = new_point_location(initial_point, control_pts_ad);

    dealii::Point<dim,double> dXsdXp;
    for (int d=0; d<dim; ++d) {
        dXsdXp[d] = new_point_ad[d].dx(0);
    }
    return dXsdXp;
}

// template<int dim>
// template<typename real>
// dealii::Point<dim,real> dXsdXp (const dealii::Point<dim,real> &initial_point, const unsigned int ctl_index) const;

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
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
    dealii::LinearAlgebra::distributed::Vector<double> vector_to_copy_into) const
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
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim,
    const dealii::LinearAlgebra::distributed::Vector<double> vector_to_copy_from)
{
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
::deform_mesh (HighOrderGrid<dim,double,dealii::LinearAlgebra::distributed::Vector<double>,dealii::DoFHandler<dim>> &high_order_grid) const
{
    dealii::LinearAlgebra::distributed::Vector<double> surface_node_displacements(high_order_grid.surface_nodes);
    auto index = high_order_grid.surface_indices.begin();
    auto node = high_order_grid.surface_nodes.begin();
    auto new_node = surface_node_displacements.begin();
    for (; index != high_order_grid.surface_indices.end(); ++index, ++node, ++new_node) {
        const dealii::types::global_dof_index global_idof_index = *index;
        const std::pair<unsigned int, unsigned int> ipoint_component = high_order_grid.global_index_to_point_and_axis.at(global_idof_index);
        const unsigned int ipoint = ipoint_component.first;
        const unsigned int component = ipoint_component.second;
        dealii::Point<dim> old_point;
        for (int d=0;d<dim;d++) {
            old_point[d] = high_order_grid.locally_relevant_surface_points[ipoint][d];
        }
        const dealii::Point<dim> new_point = new_point_location(old_point);
        *new_node = new_point[component];
    }
    surface_node_displacements.update_ghost_values();
    surface_node_displacements -= high_order_grid.surface_nodes;
    surface_node_displacements.update_ghost_values();

    // MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //     meshmover(high_order_grid, surface_node_displacements);

    MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
        meshmover( 
          *(high_order_grid.triangulation),
          high_order_grid.initial_mapping_fe_field,
          high_order_grid.dof_handler_grid,
          high_order_grid.surface_indices,
          surface_node_displacements);
    dealii::LinearAlgebra::distributed::Vector<double> volume_displacements = meshmover.get_volume_displacements();
    high_order_grid.nodes += volume_displacements;
    high_order_grid.nodes.update_ghost_values();
}


template class FreeFormDeformation<PHILIP_DIM>;

template dealii::Point<PHILIP_DIM,double> FreeFormDeformation<PHILIP_DIM>
::evaluate_ffd (const dealii::Point<PHILIP_DIM,double> &, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;

template dealii::Point<PHILIP_DIM, double> FreeFormDeformation<PHILIP_DIM>
::new_point_location(const dealii::Point<PHILIP_DIM, double>&, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;

template dealii::Tensor<1, PHILIP_DIM, double> FreeFormDeformation<PHILIP_DIM>
::get_displacement (const dealii::Point<PHILIP_DIM,double> &, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;

template std::vector<dealii::Tensor<1, PHILIP_DIM, double>> FreeFormDeformation<PHILIP_DIM>
::get_displacement (const std::vector<dealii::Point<PHILIP_DIM,double>> &, const std::vector<dealii::Point<PHILIP_DIM,double>> &) const;

} // namespace PHiLiP
