#include "free_form_deformation.h"

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
    init_msg();

    for (unsigned int ictl = 0; ictl < n_control_pts; ++ictl) {

        std::array<unsigned int,dim> ijk = global_to_grid (ictl);

        control_pts[ictl] = origin;
        for (int d=0; d<dim; ++d) {
            control_pts[ictl] += ijk[d] / (ndim_control_pts[d] - 1.0) * parallepiped_vectors[d];
        }

    }
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
template<typename real>
dealii::Point<dim,real> FreeFormDeformation<dim>::get_local_coordinates (const dealii::Point<dim,real> p) const
{
    dealii::Point<dim,real> local_coordinates;
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

    const dealii::Tensor<1,dim,real> dX = (p - origin);
    for (int d=0;d<dim;++d) {
        local_coordinates[d] = (dX * perp_vectors[d]) / (parallepiped_vectors[d] * perp_vectors[d]);
        assert(0 < local_coordinates[d] && local_coordinates[d] < 1);
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
    for (int d=0; d<dim; ++d) {
        pcout << origin << std::endl;
        pcout << origin + parallepiped_vectors[d] << std::endl;
    }
}

template class FreeFormDeformation<PHILIP_DIM>;

} // namespace PHiLiP
