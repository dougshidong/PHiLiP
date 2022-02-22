#include <type_traits>

#include <deal.II/base/exceptions.h>

// For metric Jacobian testing
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
// *****************

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_bernstein.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bernstein.h>

#include <deal.II/numerics/vector_tools.h>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/optimization/solver_bfgs.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/integrators/l2.h>

#include "high_order_grid.h"


namespace PHiLiP {

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
unsigned int HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::nth_refinement=0;

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::HighOrderGrid(
        const unsigned int max_degree,
        const std::shared_ptr<MeshType> triangulation_input)
    : max_degree(max_degree)
    , triangulation(triangulation_input)
    , dof_handler_grid(*triangulation)
    , fe_q(max_degree) // The grid must be at least p1. A p0 solution required a p1 grid.
    , fe_system(dealii::FESystem<dim>(fe_q,dim)) // The grid must be at least p1. A p0 solution required a p1 grid.
    , solution_transfer(dof_handler_grid)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_mpi);

    Assert(max_degree > 0, dealii::ExcMessage("Grid must be at least order 1."));

    if (triangulation->n_levels() == 0) {
        // Do nothing
        // Assume stuff will be read later on.
    } else {
        initialize_with_triangulation_manifold();
    }
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::initialize_with_triangulation_manifold(const bool output_mesh)
{
    allocate();
    const dealii::ComponentMask mask(dim, true);
    get_position_vector(dof_handler_grid, volume_nodes, mask);
    //get_projected_position_vector(dof_handler_grid, volume_nodes, mask);

    volume_nodes.update_ghost_values();

    ensure_conforming_mesh();

    volume_nodes.update_ghost_values();
    update_surface_nodes();
    update_mapping_fe_field();
    reset_initial_nodes();
    if (output_mesh) output_results_vtk(nth_refinement++);

    // Used to check Jacobian validity
    const unsigned int exact_jacobian_order = (max_degree-1) * dim;
    const unsigned int min_jacobian_order = 1;
    const unsigned int used_jacobian_order = std::max(exact_jacobian_order, min_jacobian_order);
    evaluate_lagrange_to_bernstein_operator(used_jacobian_order);

    auto cell = dof_handler_grid.begin_active();
    auto endcell = dof_handler_grid.end();
    // pcout << "Disabled check_valid_cells. Took too much time due to shape_grad()." << std::endl;
    for (; cell!=endcell; ++cell) {
        if (!cell->is_locally_owned())  continue;
        const bool is_invalid_cell = check_valid_cell(cell);

        if ( !is_invalid_cell ) {
            std::cout << " Poly: " << max_degree
                      << " Grid: " << nth_refinement
                      << " Cell: " << cell->active_cell_index() << " has an invalid Jacobian." << std::endl;
            // bool fixed_invalid_cell = fix_invalid_cell(cell);
            // if (fixed_invalid_cell) std::cout << "Fixed it." << std::endl;
        }
    }
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::reinit(){
    allocate();
    const dealii::ComponentMask mask(dim, true);
    get_position_vector(dof_handler_grid, volume_nodes, mask);
    volume_nodes.update_ghost_values();
    update_surface_indices();
    update_surface_nodes();
    update_mapping_fe_field();

    // Used to check Jacobian validity
    const unsigned int exact_jacobian_order = (max_degree-1) * dim;
    const unsigned int min_jacobian_order = 1;
    const unsigned int used_jacobian_order = std::max(exact_jacobian_order, min_jacobian_order);
    evaluate_lagrange_to_bernstein_operator(used_jacobian_order);

    auto cell = dof_handler_grid.begin_active();
    auto endcell = dof_handler_grid.end();
    // pcout << "Disabled check_valid_cells. Took too much time due to shape_grad()." << std::endl;
    for (; cell!=endcell; ++cell) {
        if (!cell->is_locally_owned())  continue;
        const bool is_invalid_cell = check_valid_cell(cell);

        if ( !is_invalid_cell ) {
            std::cout << " Poly: " << max_degree
                      << " Grid: " << nth_refinement
                      << " Cell: " << cell->active_cell_index() << " has an invalid Jacobian." << std::endl;
            // bool fixed_invalid_cell = fix_invalid_cell(cell);
            // if (fixed_invalid_cell) std::cout << "Fixed it." << std::endl;
        }
    }
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::ensure_conforming_mesh() {

    volume_nodes.update_ghost_values();

    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_grid, hanging_node_constraints);
    hanging_node_constraints.close();
    hanging_node_constraints.distribute(volume_nodes);

    volume_nodes.update_ghost_values();

    update_mapping_fe_field();
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::update_mapping_fe_field() {
    const dealii::ComponentMask mask(dim, true);
    mapping_fe_field = std::make_shared< dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> > (dof_handler_grid,volume_nodes,mask);
    initial_mapping_fe_field = std::make_shared< dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> > (dof_handler_grid,initial_volume_nodes,mask);
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void 
HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::allocate() 
{
    dof_handler_grid.initialize(*triangulation, fe_system);
    dof_handler_grid.distribute_dofs(fe_system);
    dealii::DoFRenumbering::Cuthill_McKee(dof_handler_grid);


    locally_owned_dofs_grid = dof_handler_grid.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_grid, locally_relevant_dofs_grid);
    ghost_dofs_grid = locally_relevant_dofs_grid;
    ghost_dofs_grid.subtract_set(locally_owned_dofs_grid);

    // reinit nodes depending on the MeshType::VectorType
    MeshTypeHelper<MeshType>::reinit_vector(
        volume_nodes, 
        dof_handler_grid, 
        locally_owned_dofs_grid,
        ghost_dofs_grid,
        mpi_communicator);
}

//template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
//dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> 
//HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::get_MappingFEField() {
//    const dealii::ComponentMask mask(dim, true);
//    dealii::VectorTools::get_position_vector(dof_handler_grid, volume_nodes, mask);
//
//    dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> mapping(dof_handler_grid,volume_nodes,mask);
//
//    return mapping;
//}
//


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>
::get_position_vector(const DoFHandlerType &dh, VectorType &position_vector, const dealii::ComponentMask &mask)
{
    AssertDimension(position_vector.size(), dh.n_dofs());
    const dealii::FESystem<dim, dim> &fe = dh.get_fe();

    // Construct default fe_mask;
    const dealii::ComponentMask fe_mask(mask.size() ? mask : dealii::ComponentMask(fe.get_nonzero_components(0).size(), true));

    AssertDimension(fe_mask.size(), fe.get_nonzero_components(0).size());

    std::vector<unsigned int> fe_to_real(fe_mask.size(), dealii::numbers::invalid_unsigned_int);
    unsigned int              size = 0;
    for (unsigned int i = 0; i < fe_mask.size(); ++i) {
        if (fe_mask[i]) fe_to_real[i] = size++;
    }
    Assert(size == dim, dealii::ExcMessage("The Component Mask you provided is invalid. It has to select exactly dim entries."));

    const dealii::Quadrature<dim> quad(fe.get_unit_support_points());

    dealii::MappingQGeneric<dim, dim> map_q(fe.degree);
    dealii::FEValues<dim, dim> fe_v(map_q, fe, quad, dealii::update_quadrature_points);
    std::vector<dealii::types::global_dof_index> dofs(fe.dofs_per_cell);

    AssertDimension(fe.dofs_per_cell, fe.get_unit_support_points().size());
    Assert(fe.is_primitive(), dealii::ExcMessage("FE is not Primitive! This won't work."));

    for (const auto &cell : dh.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            fe_v.reinit(cell);
            cell->get_dof_indices(dofs);
            const std::vector<dealii::Point<dim>> &points = fe_v.get_quadrature_points();
            for (unsigned int q = 0; q < points.size(); ++q) {
                const unsigned int comp = fe.system_to_component_index(q).first;
                if (fe_mask[comp]) ::dealii::internal::ElementAccess<VectorType>::set(points[q][fe_to_real[comp]], dofs[q], position_vector);
            }
        }
    }
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>
::get_projected_position_vector(const DoFHandlerType &dh, VectorType &position_vector, const dealii::ComponentMask &mask)
{
    AssertDimension(position_vector.size(), dh.n_dofs());
    const dealii::FESystem<dim, dim> &fe = dh.get_fe();

    // Construct default fe_mask;
    const dealii::ComponentMask fe_mask(mask.size() ? mask : dealii::ComponentMask(fe.get_nonzero_components(0).size(), true));

    AssertDimension(fe_mask.size(), fe.get_nonzero_components(0).size());

    std::vector<unsigned int> fe_to_real(fe_mask.size(), dealii::numbers::invalid_unsigned_int);
    unsigned int              size = 0;
    for (unsigned int i = 0; i < fe_mask.size(); ++i) {
        if (fe_mask[i]) fe_to_real[i] = size++;
    }
    Assert(size == dim, dealii::ExcMessage("The Component Mask you provided is invalid. It has to select exactly dim entries."));

    const dealii::QGauss<dim> quad((fe.degree+1) * 2);

    dealii::MappingQGeneric<dim, dim> map_q(fe.degree);
    dealii::FEValues<dim, dim> fe_values_curved(map_q, fe, quad, dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::MappingQ1<dim, dim> map_q1;
    dealii::FEValues<dim, dim> fe_values_straight(map_q1, fe, quad, dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_jacobians);

    const unsigned int n_dofs = fe.dofs_per_cell;
    dealii::FullMatrix<double> mass_matrix(n_dofs, n_dofs);
    dealii::FullMatrix<double> inv_mass_matrix(n_dofs, n_dofs);

    fe_values_straight.reinit(dh.begin_active());
    dealii::LocalIntegrators::L2::mass_matrix(mass_matrix, fe_values_straight);
    inv_mass_matrix.invert(mass_matrix);

    std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

    AssertDimension(fe.dofs_per_cell, fe.get_unit_support_points().size());
    Assert(fe.is_primitive(), dealii::ExcMessage("FE is not Primitive! This won't work."));

    for (const auto &cell : dh.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            fe_values_curved.reinit(cell);
            fe_values_straight.reinit(cell);
            //dealii::LocalIntegrators::L2::mass_matrix(mass_matrix, fe_values_straight);
            //mass_matrix.print(std::cout);
            //inv_mass_matrix.invert(mass_matrix);

            cell->get_dof_indices(dof_indices);
            const std::vector<dealii::Point<dim>> &quadrature_points_values = fe_values_curved.get_quadrature_points();

            dealii::Vector<double> rhs(n_dofs);
            for (unsigned int idof=0; idof<n_dofs; ++idof) {
                const unsigned int axis = fe.system_to_component_index(idof).first;
                rhs[idof] = 0.0;
                for (unsigned int q=0; q<quadrature_points_values.size(); ++q) {
                    rhs[idof] += fe_values_curved.shape_value_component(idof, q, axis) * quadrature_points_values[q][axis] * quad.weight(q);
                }
            }
            dealii::Vector<double> new_dof_values(n_dofs);
            inv_mass_matrix.vmult(new_dof_values, rhs);

            for (unsigned int idof = 0; idof < new_dof_values.size(); ++idof) {
                const unsigned int comp = fe.system_to_component_index(idof).first;
                if (fe_mask[comp]) ::dealii::internal::ElementAccess<VectorType>::set(new_dof_values[idof], dof_indices[idof], position_vector);
            }
        }
    }
}
//template <int dim, typename real>
//void HighOrderGrid<dim,real>
//::get_surface_projected_position_vector(const DoFHandlerType &dh, VectorType &position_vector, const dealii::ComponentMask &mask)
//{
//    AssertDimension(position_vector.size(), dh.n_dofs());
//    const dealii::FESystem<dim, dim> &fe = dh.get_fe();
//
//    // Construct default fe_mask;
//    const dealii::ComponentMask fe_mask(mask.size() ? mask : dealii::ComponentMask(fe.get_nonzero_components(0).size(), true));
//
//    AssertDimension(fe_mask.size(), fe.get_nonzero_components(0).size());
//
//    std::vector<unsigned int> fe_to_real(fe_mask.size(), dealii::numbers::invalid_unsigned_int);
//    unsigned int              size = 0;
//    for (unsigned int i = 0; i < fe_mask.size(); ++i) {
//        if (fe_mask[i]) fe_to_real[i] = size++;
//    }
//    Assert(size == dim, dealii::ExcMessage("The Component Mask you provided is invalid. It has to select exactly dim entries."));
//
//    const dealii::QGauss<dim> quad((fe.degree+1) * 2);
//
//    dealii::MappingQGeneric<dim, dim> map_q(fe.degree);
//    dealii::FEValues<dim, dim> fe_values_curved(map_q, fe, quad, dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
//
//    dealii::MappingQ1<dim, dim> map_q1;
//    dealii::FEValues<dim, dim> fe_values_straight(map_q1, fe, quad, dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_jacobians);
//
//    const unsigned int n_dofs = fe.dofs_per_cell;
//    dealii::FullMatrix<double> mass_matrix(n_dofs, n_dofs);
//    dealii::FullMatrix<double> inv_mass_matrix(n_dofs, n_dofs);
//
//    std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_face);
//
//    AssertDimension(fe.dofs_per_cell, fe.get_unit_support_points().size());
//    Assert(fe.is_primitive(), dealii::ExcMessage("FE is not Primitive! This won't work."));
//
//    dealii::TrilinosWrappers::SparseMatrix surface_mass_matrix;
//
//    dealii::FEFaceValues<dim,dim>  fe_face_values_curved(map_q, fe, quad, dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_jacobians);
//    dealii::FEFaceValues<dim,dim>  fe_face_values_straight(map_q1, fe, quad, dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_jacobians);
//    for (const auto &cell : dh.active_cell_iterators()) {
//
//        if (!(cell->is_locally_owned())) continue;
//
//        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
//            auto face = cell->face(iface);
//            if (!face->at_boundary()) continue;
//            const unsigned int boundary_id = face->boundary_id();
//            if (!(boundary_id == 1001)) continue;
//
//            face->get_dof_indices(dof_indices);
//
//            fe_face_values_curved.reinit(current_cell, iface);
//            fe_face_values_straight.reinit(current_cell, iface);
//
//
//
//            for (unsigned int itest=0; itest<dof_indices.size(); ++itest) {
//                const unsigned int itest_axis = fe.system_to_component_index(itest).first;
//                rhs[itest] = 0.0;
//                for (unsigned int q=0; q<quadrature_points_values.size(); ++q) {
//                    rhs[itest] += fe_values_curved.shape_value_component(itest, q, itest_axis) * quadrature_points_values[q][itest_axis] * quad.weight(q);
//                }
//            }
//
//        }
//
//        fe_values_curved.reinit(cell);
//        fe_values_straight.reinit(cell);
//        //dealii::LocalIntegrators::L2::mass_matrix(mass_matrix, fe_values_straight);
//        //mass_matrix.print(std::cout);
//        //inv_mass_matrix.invert(mass_matrix);
//
//        cell->get_dof_indices(dof_indices);
//        const std::vector<dealii::Point<dim>> &quadrature_points_values = fe_values_curved.get_quadrature_points();
//
//        dealii::Vector<double> rhs(n_dofs);
//        for (unsigned int idof=0; idof<n_dofs; ++idof) {
//            const unsigned int axis = fe.system_to_component_index(idof).first;
//            rhs[idof] = 0.0;
//            for (unsigned int q=0; q<quadrature_points_values.size(); ++q) {
//                rhs[idof] += fe_values_curved.shape_value_component(idof, q, axis) * quadrature_points_values[q][axis] * quad.weight(q);
//            }
//        }
//        dealii::Vector<double> new_dof_values(n_dofs);
//        inv_mass_matrix.vmult(new_dof_values, rhs);
//
//        for (unsigned int idof = 0; idof < new_dof_values.size(); ++idof) {
//            const unsigned int comp = fe.system_to_component_index(idof).first;
//            if (fe_mask[comp]) ::dealii::internal::ElementAccess<VectorType>::set(new_dof_values[idof], dof_indices[idof], position_vector);
//        }
//    }
//}


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
template <typename real2>
inline real2 HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>
::determinant(const std::array< dealii::Tensor<1,dim,real2>, dim > jacobian) const
{
    if constexpr(dim == 1) return jacobian[0][0];
    if constexpr(dim == 2)
        return jacobian[0][0] * jacobian[1][1] - jacobian[1][0] * jacobian[0][1];
    if constexpr(dim == 3)
        return (+ jacobian[0][0] *(jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
                - jacobian[0][1] *(jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
                + jacobian[0][2] *(jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]));
}


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
std::vector<real> HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::evaluate_jacobian_at_points(
        const VectorType &solution,
        const typename DoFHandlerType::cell_iterator &cell,
        const std::vector<dealii::Point<dim>> &points) const
{
    const dealii::FESystem<dim> &fe = cell->get_fe();
    const unsigned int n_quad_pts = points.size();
    const unsigned int n_dofs_cell = fe.n_dofs_per_cell();
    const unsigned int n_axis = dim;

    std::vector<dealii::types::global_dof_index> dofs_indices(n_dofs_cell);
    cell->get_dof_indices (dofs_indices);

    std::array<real,n_axis> coords;
    std::array< dealii::Tensor<1,dim,real>, n_axis > coords_grad;
    std::vector<real> jac_det(n_quad_pts);

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        for (unsigned int iaxis=0; iaxis<n_axis; ++iaxis) {
            coords[iaxis]      = 0;
            coords_grad[iaxis] = 0;
        }
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const unsigned int axis = fe.system_to_component_index(idof).first;
            coords[axis]      += solution[dofs_indices[idof]] * fe.shape_value_component (idof, points[iquad], axis);
            coords_grad[axis] += solution[dofs_indices[idof]] * fe.shape_grad_component (idof, points[iquad], axis);
        }
        jac_det[iquad] = determinant(coords_grad);

        if(jac_det[iquad] < 0) {
            dealii::Point<dim> phys_point;
            for (int d=0;d<dim;++d) {
                phys_point[d] = coords[d];
            }
            std::cout << " Negative Jacobian determinant. Ref Point: " << points[iquad]
                      << " Phys Point: " << phys_point
                      << " J: " << jac_det[iquad] << std::endl;
            std::cout << "Jacobian " << std::endl;
            for (int row=0;row<dim;++row) {
                for (int col=0;col<dim;++col) {
                    std::cout << coords_grad[row][col] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    return jac_det;
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
template <typename real2>
real2 HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::evaluate_jacobian_at_point(
        const std::vector<real2> &dofs,
        const dealii::FESystem<dim> &fe,
        const dealii::Point<dim> &point) const
{
    AssertDimension(dim, fe.n_components());

    const unsigned int n_dofs_coords = fe.n_dofs_per_cell();
    const unsigned int n_axis = dim;

    std::array< dealii::Tensor<1,dim,real2>, n_axis > coords_grad; // Tensor initialize with zeros
    for (unsigned int idof=0; idof<n_dofs_coords; ++idof) {
        const unsigned int axis = fe.system_to_component_index(idof).first;
        coords_grad[axis] += dofs[idof] * fe.shape_grad (idof, point);
    }
    return determinant(coords_grad);
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
template <typename real2>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::evaluate_jacobian_at_points(
        const std::vector<real2> &dofs,
        const dealii::FESystem<dim> &fe,
        const std::vector<dealii::Point<dim>> &points,
        std::vector<real2> &jacobian_determinants) const
{
    AssertDimension(dim, fe.n_components());
    AssertDimension(jacobian_determinants.size(), points.size());

    const unsigned int n_points = points.size();

    for (unsigned int i=0; i<n_points; ++i) {
        jacobian_determinants[i] = evaluate_jacobian_at_point(dofs, fe, points[i]);
    }
}

template<typename real>
std::vector<real> matrix_stdvector_mult(const dealii::FullMatrix<double> &matrix, const std::vector<real> &vector_in)
{
    const unsigned int m = matrix.m(), n = matrix.n();
    AssertDimension(vector_in.size(),n);
    std::vector<real> vector_out(m,0.0);
    for (unsigned int row=0; row<m; ++row) {
        for (unsigned int col=0; col<n; ++col) {
            vector_out[row] += matrix[row][col]*vector_in[col];
        }
    }
    return vector_out;
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::evaluate_lagrange_to_bernstein_operator(const unsigned int order)
{
    const dealii::FE_Q<dim> lagrange_basis(order);
    const std::vector< dealii::Point<dim> > &lagrange_pts = lagrange_basis.get_unit_support_points();
    const unsigned int n_lagrange_pts = lagrange_pts.size();

    const dealii::FE_Bernstein<dim> bernstein_basis(order);
    const unsigned int n_bernstein = bernstein_basis.n_dofs_per_cell();
    AssertDimension(n_bernstein, n_lagrange_pts);
    // Evaluate Vandermonde matrix such that V u_bernstein = u_lagrange
    // where the matrix's rows correspond to the different Bernstein polynomials
    // and the matrix's column correspond to the unit support points of the Lagrage polynomials
    dealii::FullMatrix<double> bernstein_to_lagrange(n_bernstein, n_lagrange_pts);
    for (unsigned int ibernstein=0; ibernstein<n_bernstein; ++ibernstein) {
        for (unsigned int ijacpts=0; ijacpts<n_bernstein; ++ijacpts) {
            const dealii::Point<dim> &point = lagrange_pts[ijacpts];
            bernstein_to_lagrange[ibernstein][ijacpts] = bernstein_basis.shape_value(ibernstein, point);
        }
    }
    lagrange_to_bernstein_operator.reinit(n_lagrange_pts, n_bernstein);
    if (n_lagrange_pts > 1000) pcout << "Careful, about to invert a " << n_lagrange_pts << " x " << n_lagrange_pts << " dense matrix..." << std::endl;
    lagrange_to_bernstein_operator.invert(bernstein_to_lagrange);
    if (n_lagrange_pts > 1000) pcout << "Done inverting a " << n_lagrange_pts << " x " << n_lagrange_pts << " dense matrix..." << std::endl;
}


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
bool HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::check_valid_cell(const typename DoFHandlerType::cell_iterator &cell) const
{
    return true;
    const unsigned int exact_jacobian_order = (max_degree-1) * dim, min_jacobian_order = 1;
    const unsigned int used_jacobian_order = std::max(exact_jacobian_order, min_jacobian_order);

    // Evaluate Jacobian at Lagrange interpolation points
    const dealii::FESystem<dim> &fe_coords = cell->get_fe();
    const unsigned int n_dofs_coords = fe_coords.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> dofs_indices(n_dofs_coords);
    cell->get_dof_indices (dofs_indices);

    std::vector< real > cell_nodes(n_dofs_coords);
    for (unsigned int idof = 0; idof < n_dofs_coords; ++idof) {
        cell_nodes[idof] = volume_nodes(dofs_indices[idof]);
    }

    const dealii::FE_Q<dim> lagrange_basis(used_jacobian_order);
    const std::vector< dealii::Point<dim> > &lagrange_pts = lagrange_basis.get_unit_support_points();
    const unsigned int n_lagrange_pts = lagrange_pts.size();
    const unsigned int n_bernstein = n_lagrange_pts;
    std::vector<real> lagrange_coeff(n_lagrange_pts);
    evaluate_jacobian_at_points(cell_nodes, fe_coords, lagrange_pts, lagrange_coeff);
    std::vector<real> bernstein_coeff = matrix_stdvector_mult(lagrange_to_bernstein_operator, lagrange_coeff);

    const real tol = 1e-12;
    for (unsigned int i=0; i<n_bernstein;++i) {
        if (bernstein_coeff[i] <= tol) {
            std::cout << "Negative bernstein coeff: " << i << " " << bernstein_coeff[i] << std::endl;
            // std::cout << "Bernstein vector: " ;
            // for (unsigned int j=0; j<n_bernstein;++j) {
            //     std::cout << bernstein_coeff[j] << " ";
            // }
            //std::cout << std::endl;
            return false;
        }
    }
    return true;
}

// dealii::FullMatrix<double> lagrange_to_bernstein_operator(
//     const dealii::FE_Q<dim> &lagrange_basis,
//     const dealii::FE_Bernstein<dim> &bernstein_basis,
//     const typename DoFHandlerType::cell_iterator &cell)
// {
//     const dealii::FE_Bernstein<dim> bernstein_basis(jacobian_order);
//     const unsigned int n_bernstein = bernstein_basis.n_dofs_per_cell();
//     const unsigned int n_lagrange_pts = lagrange_pts.size();
//     AssertDimension(n_bernstein, n_lagrange_pts);
//     // Evaluate Vandermonde matrix such that V u_bernstein = u_lagrange
//     // where the matrix's rows correspond to the different Bernstein polynomials
//     // and the matrix's column correspond to the unit support points of the Lagrage polynomials
//     dealii::FullMatrix<double> bernstein_to_lagrange(n_bernstein, n_lagrange_pts);
//     for (unsigned int ibernstein=0; ibernstein<n_bernstein; ++ibernstein) {
//         for (unsigned int ijacpts=0; ijacpts<n_bernstein; ++ijacpts) {
//             const dealii::Point<dim> &point = lagrange_pts[ijacpts];
//             bernstein_to_lagrange[ibernstein][ijacpts] = bernstein_basis.shape_value(ibernstein, point);
//         }
//     }
// }

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
bool HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::fix_invalid_cell(const typename DoFHandlerType::cell_iterator &cell)
{
    // This will be the target ratio between the current minimum (estimated) cell Jacobian
    // and the value of the straight-sided Jacobian. This estimates depends on the Bernstein
    // coefficients that serve as a convex hull
    const double target_ratio = 0.1;

    // Maximum number of times we will move the barrier
    const int max_barrier_iterations = 100;

    const unsigned int exact_jacobian_order = (max_degree-1) * dim, min_jacobian_order = 1;
    const unsigned int used_jacobian_order = std::max(exact_jacobian_order, min_jacobian_order);
    const dealii::FE_Q<dim> lagrange_basis(used_jacobian_order);
    const std::vector< dealii::Point<dim> > &lagrange_pts = lagrange_basis.get_unit_support_points();
    const unsigned int n_lagrange_pts = lagrange_pts.size(), n_bernstein = n_lagrange_pts;

    const dealii::FESystem<dim> &fe_coords = cell->get_fe();
    const unsigned int n_dofs_coords = fe_coords.n_dofs_per_cell();
    // Evaluate Jacobian at Lagrange interpolation points
    std::vector<dealii::types::global_dof_index> dofs_indices(n_dofs_coords);
    cell->get_dof_indices (dofs_indices);

    // Use reverse mode for more efficiency
    using FadType = Sacado::Fad::DFad<real>;
    std::vector<FadType> cell_nodes(n_dofs_coords);
    std::vector<FadType> lagrange_coeff(n_lagrange_pts);
    std::vector<FadType> bernstein_coeff(n_bernstein);

    // Count and tag movable volume_nodes
    std::vector< bool > movable(n_dofs_coords);
    unsigned int n_movable_nodes = 0;
    // for (unsigned int idof = 0; idof < n_dofs_coords; ++idof) {
    //     bool is_interior_node = true;
    //     for (unsigned int iface = 0; iface<dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
    //         is_interior_node = is_interior_node && !(fe_coords.has_support_on_face(idof, iface));
    //     }
    //     movable[idof] = is_interior_node;
    //     if (is_interior_node) n_movable_nodes++;
    // }
    for (unsigned int idof = 0; idof < n_dofs_coords; ++idof) {
        const bool is_movable = (idof/dim > 2*dim);
        movable[idof] = is_movable;
        if (is_movable) n_movable_nodes++;
    }

    // Evaluate straight sided cell Jacobian.
    // Note that we can't simply use the Triangulation's cell's vertices since
    // MappingFEField does not preserves_vertex_locations()
    const unsigned int n_vertices = dealii::GeometryInfo<dim>::vertices_per_cell;
    const dealii::FE_Q<dim> straight_sided_elem_fe(1);
    const dealii::FESystem<dim> straight_sided_elem_fesystem(straight_sided_elem_fe, dim);
    std::vector<real> straight_sided_dofs(straight_sided_elem_fesystem.n_dofs_per_cell());
    for (unsigned int ivertex = 0; ivertex < n_vertices; ++ivertex) {
        const dealii::Point<dim> unit_vertex = dealii::GeometryInfo<dim>::unit_cell_vertex(ivertex);
        const dealii::Point<dim> phys_vertex = mapping_fe_field->transform_unit_to_real_cell(cell, unit_vertex);
        for (int d=0;d<dim;++d) {
            straight_sided_dofs[ivertex*dim+d] = phys_vertex[d];
        }
    }
    dealii::Point<dim> unit_cell_center;
    unit_cell_center[0] = 0.5;
    if constexpr (dim>=2) unit_cell_center[1] = 0.5;
    if constexpr (dim>=3) unit_cell_center[2] = 0.5;

    //const double straight_sided_jacobian = cell->measure();
    const double straight_sided_jacobian = evaluate_jacobian_at_point(straight_sided_dofs, straight_sided_elem_fesystem, unit_cell_center);


    // Initialize movable volume_nodes, which are our design variables
    dealii::Vector<real> movable_nodes(n_movable_nodes);
    unsigned int idesign = 0;
    for (unsigned int idof = 0; idof < n_dofs_coords; ++idof) {
        cell_nodes[idof] = volume_nodes(dofs_indices[idof]);
        if (movable[idof]) movable_nodes[idesign++] = cell_nodes[idof].val();
    }
    evaluate_jacobian_at_points(cell_nodes, fe_coords, lagrange_pts, lagrange_coeff);
    bernstein_coeff = matrix_stdvector_mult(lagrange_to_bernstein_operator, lagrange_coeff);
    real min_ratio = -1.0;
    for (int barrier_iterations = 0; barrier_iterations < max_barrier_iterations && min_ratio < target_ratio; ++barrier_iterations) {
        min_ratio = bernstein_coeff[0].val();
        for (unsigned int i=0; i<n_bernstein;++i) {
            min_ratio = std::min(bernstein_coeff[i].val(), min_ratio);
        }
        min_ratio /= straight_sided_jacobian;
        std::cout << " Barrier iteration : " << barrier_iterations << std::endl;
        std::cout << "Current minimum Jacobian ratio: " << min_ratio << std::endl;

        const double barrier = min_ratio - 0.10*std::abs(min_ratio);
        const double barrierJac = barrier*straight_sided_jacobian;

        std::function<real (const dealii::Vector<real> &, dealii::Vector<real> &)> func =
            [&](const dealii::Vector<real> &movable_nodes, dealii::Vector<real> &gradient) {

            unsigned int idesign = 0;
            for (unsigned int idof = 0; idof < n_dofs_coords; ++idof) {
                if (movable[idof]) {
                    cell_nodes[idof] = movable_nodes[idesign];
                    cell_nodes[idof].diff(idesign, n_movable_nodes);
                    idesign++;
                }
            }
            evaluate_jacobian_at_points(cell_nodes, fe_coords, lagrange_pts, lagrange_coeff);
            bernstein_coeff = matrix_stdvector_mult(lagrange_to_bernstein_operator, lagrange_coeff);

            FadType functional = 0.0;
            min_ratio = bernstein_coeff[0].val();
            for (unsigned int i=0; i<n_bernstein;++i) {
                FadType logterm = std::log((bernstein_coeff[i] - barrierJac) / (straight_sided_jacobian - barrierJac));
                FadType quadraticterm = bernstein_coeff[i] / straight_sided_jacobian - 1.0;
                functional += std::pow(logterm,2) + std::pow(quadraticterm,2);
                min_ratio = std::min(bernstein_coeff[i].val(), min_ratio);
            }
            min_ratio /= straight_sided_jacobian;
            //FadType functional = bernstein_coeff[0];
            //for (unsigned int i=0; i<n_bernstein;++i) {
            //    functional = std::min(functional, bernstein_coeff[i]);
            //}
            //functional *= -1.0;

            for (unsigned int i = 0; i < movable_nodes.size(); ++i) {
                //gradient[i] = functional.fastAccessDx(i);
                gradient[i] = functional.dx(i);
            }
            return functional.val();
        };

        // const unsigned int bfgs_max_it = 150;
        // const double gradient_tolerance = 1e-5;
        // const unsigned int max_history = bfgs_max_it;

        // dealii::SolverControl solver_control(bfgs_max_it, gradient_tolerance, false);
        // typename dealii::SolverBFGS<dealii::Vector<real>>::AdditionalData data(max_history, false);
        // dealii::SolverBFGS<dealii::Vector<real>> solver(solver_control, data);
        // solver.connect_preconditioner_slot(preconditioner);
        // solver.solve(func, movable_nodes);
        const unsigned int max_opt_iterations = 1000;
        const unsigned int n_line_search = 40;
        const double initial_step_length = 1.0;//1e-3 * cell->minimum_vertex_distance();
        dealii::Vector<real> old_movable_nodes(n_movable_nodes);
        dealii::Vector<real> search_direction(n_movable_nodes);
        dealii::Vector<real> grad(n_movable_nodes);
        dealii::Vector<real> old_grad(n_movable_nodes);
        real functional = func(movable_nodes, grad);
        const real initial_grad_norm = grad.l2_norm();
        double grad_norm = grad.l2_norm();
        dealii::FullMatrix<real> hessian_inverse(n_movable_nodes);
        dealii::FullMatrix<real> outer_product_term(n_movable_nodes);
        dealii::Vector<real> dg(n_movable_nodes);
        dealii::Vector<real> dx(n_movable_nodes);
        dealii::Vector<real> B_dg(n_movable_nodes);
        hessian_inverse = 0;
        for (unsigned int inode=0; inode<n_movable_nodes; ++inode) {
            hessian_inverse[inode][inode] = 1.0e-8;
        }


        const double gradient_drop = 1e-2;
        for (unsigned int i=0;i<max_opt_iterations && grad_norm/initial_grad_norm > gradient_drop;++i) {

            old_movable_nodes = movable_nodes;
            old_grad = grad;

            hessian_inverse.vmult(search_direction, grad);
            search_direction *= -1.0;
            double step_length = initial_step_length;
            real old_functional = functional;
            unsigned int i_line_search;
            for (i_line_search = 0; i_line_search<n_line_search; ++i_line_search) {
                movable_nodes = old_movable_nodes;
                movable_nodes.add(step_length, search_direction);
                try {
                    functional = func(movable_nodes, grad);
                    if (std::isnan(functional)) throw -1;
                    if (functional-old_functional  < 1e-4 * step_length * (grad*search_direction)) break; // Armijo condition satisfied.
                } catch (...)
                {}
                step_length *= 0.5;
            }
            //if (i_line_search == n_line_search) {
            //    hessian_inverse = 0;
            //    for (unsigned int inode=0; inode<n_movable_nodes; ++inode) {
            //        hessian_inverse[inode][inode] = 1.0e-8;
            //    }
            //    hessian_inverse.vmult(search_direction, grad);
            //    search_direction *= -1.0;
            //    double step_length = initial_step_length;
            //    real old_functional = functional;
            //    unsigned int i_line_search;
            //    for (i_line_search = 0; i_line_search<n_line_search; ++i_line_search) {
            //        movable_nodes = old_movable_nodes;
            //        movable_nodes.add(step_length, search_direction);
            //        try {
            //            functional = func(movable_nodes, grad);
            //            if (std::isnan(functional)) throw -1;
            //            if (functional-old_functional  < 1e-4 * step_length * (grad*search_direction)) break; // Armijo condition satisfied.
            //        } catch (...)
            //        {}
            //        step_length *= 0.5;
            //    }
            //}
            grad_norm = grad.l2_norm();
            std::cout << " Barrier its : " << barrier_iterations
                      << " min_ratio: " << min_ratio
                      << " BFGS its : " << i
                      << " Func : " << functional
                      << " |Grad|: " << grad_norm / initial_grad_norm
                      << " Step length: " << step_length
                      << std::endl;
            // BFGS inverse Hessian update
            dx = movable_nodes; dx -= old_movable_nodes;
            dg = grad;          dg -= old_grad;
            const real dgdx = dg*dx;
            if (dgdx < 0) continue; // skip bfgs update if negative curvature
            hessian_inverse.vmult(B_dg, dg);
            const real dg_B_dg = dg*B_dg;
            const real scaling1 = (dgdx + dg_B_dg) / (dgdx*dgdx);
            const real scaling2 = -1.0/dgdx;
            outer_product_term.outer_product(dx, dx);
            hessian_inverse.add(scaling1, outer_product_term);
            outer_product_term.outer_product(B_dg, dx);
            hessian_inverse.add(scaling2, outer_product_term);
            hessian_inverse.Tadd(scaling2, outer_product_term);
        }
    }


    const real tol = 1e-12;
    for (unsigned int i=0; i<n_bernstein;++i) {
        if (bernstein_coeff[i] <= tol) {
            std::cout << "Unable to fix cell "<< std::endl;
            std::cout << "Bernstein coeff: " << bernstein_coeff[i] << std::endl;
            std::cout << "Bernstein vector: " ;
            for (unsigned int j=0; j<n_bernstein;++j) {
                std::cout << bernstein_coeff[j] << " ";
            }
            std::cout << std::endl;
            return false;
        }
    }
    return true;
}



// template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
// bool HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::make_valid_cell(
//         const typename DoFHandlerType::cell_iterator &cell);
// {
//     const dealii::FESystem<dim> &fe_coords = cell->get_fe();
//     const order = fe_coords.tensor_degree();
//     const jacobian_order = std::pow(fe_coords.tensor_degree()-1, dim);
//     const dealii::FE_Q<dim> lagrange_basis(jacobian_order);
//     const std::vector< dealii::Point<dim> > &jacobian_points = lagrange_basis.get_unit_support_points();
//
//     const dealii::FE_Bernstein<dim> bernstein_basis(jacobian_order);
//
//     const unsigned int n_lagrange_pts = jacobian_points.size();
//     const unsigned int n_dofs_coords = fe_coords.n_dofs_per_cell();
//     const unsigned int n_axis = dim;
//
//     // Evaluate Jacobian at Lagrange interpolation points
//     std::vector<dealii::types::global_dof_index> dofs_indices(n_dofs_cell);
//     cell->get_dof_indices (dofs_indices);
//
//     dealii::Vector<double> lagrange_coeff(n_lagrange_pts);
//
//     for (unsigned int i_lagrange=0; i_lagrange<n_lagrange_pts; ++i_lagrange) {
//
//         const dealii::Point<dim> &point = jacobian_points[i_lagrange];
//
//         std::array< dealii::Tensor<1,dim,real>, n_axis > > coords_grad; // Tensor initialize with zeros
//         for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
//             const unsigned int axis = fe.system_to_component_index(idof).first;
//             coords_grad[axis] += coords[dofs_indices[idof]] * fe.shape_grad (idof, point);
//         }
//         dealii::Tensor<2,dim,real> jacobian;
//         for (unsigned int a=0;a<n_axis;++a) {
//             for (int d=0;d<dim;++d) {
//                 jacobian[a][d] = coords_grad[iquad][a][d];
//             }
//         }
//         dealii::Vector<double> lagrange_coeff(i_lagrange);
//     }
//
//     const unsigned int n_bernstein = bernstein_basis.n_dofs_per_cell();
//     AssertDimension(n_bernstein == n_lagrange_pts);
//     // Evaluate Vandermonde matrix such that V u_bernstein = u_lagrange
//     // where the matrix's rows correspond to the different Bernstein polynomials
//     // and the matrix's column correspond to the unit support points of the Lagrage polynomials
//     dealii::FullMatrix<double> bernstein_to_lagrange(n_bernstein, n_lagrange_pts);
//     for (unsigned int ibernstein=0; ibernstein<n_bernstein; ++ibernstein) {
//         for (unsigned int ijacpts=0; ijacpts<n_bernstein; ++ijacpts) {
//             const dealii::Point<dim> &point = jacobian_points[ijacpts];
//             bernstein_to_lagrange[ibernstein][ijacpts] = bernstein_basis.shape_value(ibernstein, point);
//         }
//     }
//     dealii::FullMatrix<double> lagrange_to_bernstein;
//     lagrange_to_bernstein.invert(bernstein_to_lagrange);
//
//     dealii::Vector<double> bernstein_coeff(n_bernstein);
//     lagrange_to_bernstein.vmult(bernstein_coeff, lagrange_coeff);
//
//     return false;
// }

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::test_jacobian()
{
    // // Setup a dummy system
    // const unsigned int solution_degree = max_degree-1;
    // const unsigned int dummy_n_state = 5;
    // const dealii::FE_DGQ<dim> fe_dgq(solution_degree);
    // const dealii::FESystem<dim> fe_system_dgq(fe_dgq, dummy_n_state);
    // const dealii::QGauss<dim> qgauss(solution_degree+1);
    // const dealii::QGaussLobatto<dim> qgauss_lobatto(solution_degree+20);

    // dealii::DoFHandler<dim> dof_handler;
    // dof_handler.initialize(*triangulation, fe_system_dgq);

    // const dealii::Mapping<dim> &mapping = (*(mapping_fe_field));
    // const dealii::FESystem<dim> &fe = fe_system_dgq;
    // const dealii::Quadrature<dim> &quadrature = qgauss;
    // const dealii::UpdateFlags volume_update_flags =
    //     dealii::update_values
    //     | dealii::update_gradients
    //     | dealii::update_quadrature_points
    //     | dealii::update_JxW_values
    //     | dealii::update_jacobians
    //     ;

    // {
    //     const dealii::Quadrature<dim> &overquadrature = qgauss_lobatto;
    //     const std::vector< dealii::Point< dim > > points = overquadrature.get_points();
    //     std::vector<real> jac_det;
    //     for (auto cell = dof_handler_grid.begin_active(); cell!=dof_handler_grid.end(); ++cell) {
    //         if (!cell->is_locally_owned()) continue;
    //         jac_det = evaluate_jacobian_at_points(volume_nodes, cell, points);
    //     }
    // }

    // dealii::FEValues<dim,dim> fe_values(mapping, fe, quadrature, volume_update_flags);

    // for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell) {

    //     if (!cell->is_locally_owned()) continue;

    //     // const unsigned int n_quad_pts = quadrature.size();

    //     // std::cout << " Cell " << cell->active_cell_index();
    //     // fe_values.reinit(cell);
    //     // for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //     //     const std::vector<dealii::Point<dim>> &points = fe_values.get_quadrature_points();
    //     //     std::cout << " Point: " << points[iquad]
    //     //               << " JxW: " << fe_values.JxW(iquad)
    //     //               << " J: " << (fe_values.jacobian(iquad)).determinant()
    //     //               << std::endl;
    //     // }

    //     // for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
    //     //     const unsigned int istate_trial = fe_values.get_fe().system_to_component_index(itrial).first;
    //     //     real value = 0.0;
    //     //     for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //     //         value += fe_values.shape_value_component(itrial,iquad,istate_trial) * fe_values.JxW(iquad);
    //     //     }
    //     // }

    //     //dofs_indices.resize(n_dofs_cell);
    //     //cell->get_dof_indices (dofs_indices);
    // }
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::refine_global()
{
    prepare_for_coarsening_and_refinement();
    triangulation->refine_global();
    execute_coarsening_and_refinement();
}


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::prepare_for_coarsening_and_refinement() {

    old_volume_nodes = volume_nodes;
    old_volume_nodes.update_ghost_values();
    if constexpr (PHILIP_DIM==1) solution_transfer.clear();
    solution_transfer.prepare_for_coarsening_and_refinement(old_volume_nodes);
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::execute_coarsening_and_refinement(const bool output_mesh) {
    allocate();
    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>,
                                 decltype(solution_transfer)>){
        solution_transfer.interpolate(old_volume_nodes, volume_nodes);
    } else {
        solution_transfer.interpolate(volume_nodes);
    }
    const bool use_manifold_for_refined_nodes = false;
    const dealii::ComponentMask mask(dim, true);
    if (use_manifold_for_refined_nodes) {
        get_position_vector(dof_handler_grid, volume_nodes, mask);
    }

    volume_nodes.update_ghost_values();

    ensure_conforming_mesh();

    update_surface_nodes();
    update_mapping_fe_field();
    if (output_mesh) output_results_vtk(nth_refinement++);

    //auto cell = dof_handler_grid.begin_active();
    //auto endcell = dof_handler_grid.end();
    //// pcout << "Disabled check_valid_cells. Took too much time due to shape_grad()." << std::endl;
    //for (; cell!=endcell; ++cell) {
    //    if (!cell->is_locally_owned())  continue;
    //    const bool is_invalid_cell = check_valid_cell(cell);

    //    if ( !is_invalid_cell ) {
    //        std::cout << " Poly: " << max_degree
    //                  << " Grid: " << nth_refinement
    //                  << " Cell: " << cell->active_cell_index() << " has an invalid Jacobian." << std::endl;
    //        //bool fixed_invalid_cell = fix_invalid_cell(cell);
    //        //if (fixed_invalid_cell) std::cout << "Fixed it." << std::endl;
    //    }
    //}
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
dealii::Point<dim> HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::smallest_cell_coordinates()
{
    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const dealii::Point<dim> unit_vertex = dealii::GeometryInfo<dim>::unit_cell_vertex(0);
    double current_cell_diameter;
    double min_diameter_local = dof_handler_grid.begin_active()->diameter();
    dealii::Point<dim> smallest_cell_coord; 

    for (auto cell = dof_handler_grid.begin_active(); cell!= dof_handler_grid.end(); ++cell) 
    {
        current_cell_diameter = cell->diameter(); // For future dealii version: current_cell_diameter = cell->diameter(*(mapping_fe_field));
        if ((min_diameter_local > current_cell_diameter) && (cell->is_locally_owned()))
        {
            min_diameter_local = current_cell_diameter;
            smallest_cell_coord = mapping_fe_field->transform_unit_to_real_cell(cell, unit_vertex);
        }
    }
    
    dealii::Utilities::MPI::MinMaxAvg minindexstore;
    minindexstore = dealii::Utilities::MPI::min_max_avg(min_diameter_local, mpi_communicator); 

    int n_proc_small = minindexstore.min_index; // Processor containing smallest diameter
    double global_point[dim];

    if (iproc == n_proc_small)
    {
       for (int i=0; i<dim; i++)
            global_point[i] = smallest_cell_coord[i];
    }
    
    MPI_Bcast(global_point, dim, MPI_DOUBLE, n_proc_small, mpi_communicator); // Update values in all processors
    
    for (int i=0; i<dim; i++)
        smallest_cell_coord[i] = global_point[i];

    return smallest_cell_coord;

}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
    std::size_t total_size = 0;
    for (const auto& sub : v) {
        total_size += sub.size();
    }
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v) {
        result.insert(result.end(), sub.begin(), sub.end());
    }
    return result;
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::update_surface_nodes() {

    update_surface_indices();

    {
        const unsigned int n_locally_owned_surface_nodes = locally_owned_surface_nodes_indices.size();
        // Copy local surface node locations
        locally_owned_surface_nodes.clear();
        locally_owned_surface_nodes.resize(n_locally_owned_surface_nodes);
        unsigned int i = 0;
        for (auto index = locally_owned_surface_nodes_indices.begin(); index != locally_owned_surface_nodes_indices.end(); index++) {
            locally_owned_surface_nodes[i++] = volume_nodes[*index];
        }

        n_locally_owned_surface_nodes_per_mpi.clear();
        n_locally_owned_surface_nodes_per_mpi.resize(n_mpi);
        MPI_Allgather(&n_locally_owned_surface_nodes, 1, MPI::UNSIGNED, &(n_locally_owned_surface_nodes_per_mpi[0]), 1, MPI::UNSIGNED, MPI_COMM_WORLD);

        std::vector<std::vector<real>> vector_locally_owned_surface_nodes(n_mpi);
        std::vector<std::vector<unsigned int>> vector_locally_owned_surface_indices(n_mpi);
        std::vector<MPI_Request> request(n_mpi);
        for (int i_mpi=0; i_mpi<n_mpi; ++i_mpi) {
            vector_locally_owned_surface_nodes[i_mpi].resize(n_locally_owned_surface_nodes_per_mpi[i_mpi]);
            vector_locally_owned_surface_indices[i_mpi].resize(n_locally_owned_surface_nodes_per_mpi[i_mpi]);
            if (i_mpi == mpi_rank) {
                vector_locally_owned_surface_nodes[i_mpi] = locally_owned_surface_nodes;
                vector_locally_owned_surface_indices[i_mpi] = locally_owned_surface_nodes_indices;
            }
        }

        for (int i_mpi=0; i_mpi<n_mpi; ++i_mpi) {
            MPI_Bcast(&(vector_locally_owned_surface_nodes[i_mpi][0]), n_locally_owned_surface_nodes_per_mpi[i_mpi], MPI_DOUBLE, i_mpi, MPI_COMM_WORLD);
            MPI_Bcast(&(vector_locally_owned_surface_indices[i_mpi][0]), n_locally_owned_surface_nodes_per_mpi[i_mpi], MPI::UNSIGNED, i_mpi, MPI_COMM_WORLD);
        }

        all_surface_nodes = flatten(vector_locally_owned_surface_nodes);
        all_surface_indices = flatten(vector_locally_owned_surface_indices);

        unsigned int low_range = 0;
        for (int i_mpi=0; i_mpi<mpi_rank; ++i_mpi) {
            low_range += n_locally_owned_surface_nodes_per_mpi[i_mpi];
        }
        const unsigned int high_range = low_range + n_locally_owned_surface_nodes_per_mpi[mpi_rank];

        const unsigned int n_surface_nodes = all_surface_nodes.size();
        locally_owned_surface_nodes_indexset.clear();
        locally_owned_surface_nodes_indexset.set_size(n_surface_nodes);
        locally_owned_surface_nodes_indexset.add_range(low_range, high_range);

    }

    {
        const unsigned int n_locally_relevant_surface_nodes = locally_relevant_surface_nodes_indices.size();
        // Copy local surface node locations
        locally_relevant_surface_nodes.clear();
        locally_relevant_surface_nodes.resize(n_locally_relevant_surface_nodes);
        unsigned int i = 0;
        for (auto index = locally_relevant_surface_nodes_indices.begin(); index != locally_relevant_surface_nodes_indices.end(); index++) {
            locally_relevant_surface_nodes[i++] = volume_nodes[*index];
        }

        std::vector<unsigned int> n_locally_relevant_surface_nodes_per_mpi(n_mpi);
        MPI_Allgather(&n_locally_relevant_surface_nodes, 1, MPI::UNSIGNED, &(n_locally_relevant_surface_nodes_per_mpi[0]), 1, MPI::UNSIGNED, MPI_COMM_WORLD);

    }

    // Find ghost_surface_nodes_indexset for the surface_nodes vector.
    {
        const unsigned int n_surface_nodes = all_surface_nodes.size();
        ghost_surface_nodes_indexset.clear();
        ghost_surface_nodes_indexset.set_size(n_surface_nodes);
        for (auto index = locally_relevant_surface_nodes_indices.begin(); index != locally_relevant_surface_nodes_indices.end(); ++index) {
            auto it = std::find (locally_owned_surface_nodes_indices.begin(), locally_owned_surface_nodes_indices.end(), *index);
            if (it == locally_owned_surface_nodes_indices.end()) {
                // If not in locally_owned_surface_nodes_indexset then, it must be a ghost entry
                bool found = false;
                for (unsigned int i = 0; i < all_surface_indices.size(); i++) {
                    if (all_surface_indices[i] == *index) {
                        ghost_surface_nodes_indexset.add_index(i);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cout << "Could not find index " << *index << " in the surface indices... Aborting. " << std::endl;
                    std::cout << "All surface_indices: " << std::endl;
                    for (unsigned int i = 0; i < all_surface_indices.size(); i++) {
                        std::cout << "i " << i << " all_surface_indices[i] " << all_surface_indices[i] << std::endl;
                    }
                    std::cout << "Locally relevant surface nodes indices: " << std::endl;
                    for (unsigned int i = 0; i < locally_relevant_surface_nodes_indices.size(); i++) {
                        std::cout << "i " << i << " locally_relevant_surface_nodes_indices[i] " << locally_relevant_surface_nodes_indices[i] << std::endl;
                    }
                    std::cout << "Locally owned surface nodes indices: " << std::endl;
                    for (unsigned int i = 0; i < locally_owned_surface_nodes_indices.size(); i++) {
                        std::cout << "i " << i << " locally_owned_surface_nodes_indices[i] " << locally_owned_surface_nodes_indices[i] << std::endl;
                    }

                    std::abort();
                }
            }
        }
    }

    surface_nodes.reinit(locally_owned_surface_nodes_indexset, ghost_surface_nodes_indexset, MPI_COMM_WORLD);
    surface_to_volume_indices.reinit(locally_owned_surface_nodes_indexset, ghost_surface_nodes_indexset, MPI_COMM_WORLD);
    unsigned int i = 0;
    auto index = surface_to_volume_indices.begin();
    AssertDimension(locally_owned_surface_nodes_indexset.n_elements(), locally_owned_surface_nodes.size());
    AssertDimension(locally_owned_surface_nodes_indexset.n_elements(), locally_owned_surface_nodes_indices.size());
    for (auto node = surface_nodes.begin(); node != surface_nodes.end(); node++, index++, i++) {
        *node = locally_owned_surface_nodes[i];
        *index = locally_owned_surface_nodes_indices[i];
    }
    surface_to_volume_indices.update_ghost_values();
    surface_nodes.update_ghost_values();

    update_map_nodes_surf_to_vol();
}


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::update_map_nodes_surf_to_vol()
{
    const unsigned int n_rows = volume_nodes.size();
    const unsigned int n_cols = surface_nodes.size();

    const dealii::IndexSet &row_part = volume_nodes.get_partitioner()->locally_owned_range();
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_grid, locally_relevant_dofs);

    const dealii::IndexSet &col_part = surface_nodes.get_partitioner()->locally_owned_range();

    dealii::DynamicSparsityPattern dsp(n_rows, n_cols, row_part);
    for (unsigned int i_col = 0; i_col < n_cols; ++i_col) {
        if (col_part.is_element(i_col)) {
            const unsigned int i_row = surface_to_volume_indices[i_col];
            dsp.add(i_row, i_col);
        }
    }

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, row_part, mpi_communicator, locally_relevant_dofs);

    map_nodes_surf_to_vol.reinit(row_part, col_part, dsp, mpi_communicator);

    for (unsigned int i_col = 0; i_col < n_cols; ++i_col) {
        if (col_part.is_element(i_col)) {
            const unsigned int i_row = surface_to_volume_indices[i_col];
            map_nodes_surf_to_vol.set(i_row, i_col, 1.0);
        }
    }
    map_nodes_surf_to_vol.compress(dealii::VectorOperation::insert);
}


template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::reset_initial_nodes()
{
    initial_volume_nodes = volume_nodes;
    initial_volume_nodes.update_ghost_values();
    initial_surface_nodes = surface_nodes;
    initial_surface_nodes.update_ghost_values();
    initial_locally_relevant_surface_points = locally_relevant_surface_points;
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::update_surface_indices() {
    locally_owned_surface_nodes_indices.clear();
    locally_owned_surface_nodes.clear();
    locally_relevant_surface_nodes_indices.clear();
    locally_relevant_surface_nodes.clear();
    locally_relevant_surface_nodes_boundary_id.clear();
    locally_relevant_surface_points.clear();
    global_index_to_point_and_axis.clear();
    point_and_axis_to_global_index.clear();

    const unsigned int dofs_per_cell = fe_system.n_dofs_per_cell();
    const unsigned int dofs_per_face = fe_system.n_dofs_per_face();
    std::vector< dealii::types::global_dof_index > dof_indices(dofs_per_cell);

    // Use unordered sets which uses hashmap.
    // Has an average complexity of O(1) (worst case O(n)) for finding and inserting
    // Overall algorithm average will be O(n), worst case O(n^2)
    std::unordered_set<dealii::types::global_dof_index> locally_relevant_surface_dof_indices_set;
    std::unordered_set<dealii::types::global_dof_index> locally_owned_surface_dof_indices_set;
    for (auto cell = dof_handler_grid.begin_active(); cell!=dof_handler_grid.end(); ++cell) {

        if (!cell->is_locally_owned() && !cell->is_ghost()) continue;
        if (!cell->at_boundary()) continue;

        cell->get_dof_indices(dof_indices);

        // Each time a new shape_within_base is found, add it to the list of points.
        std::unordered_set<dealii::types::global_dof_index> shape_within_base_found;
        std::map<unsigned int, unsigned int> shape_function_within_base_to_point_index;

        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            const auto face = cell->face(iface);

            if (!face->at_boundary()) continue;

            const unsigned int boundary_id = face->boundary_id();
            // Can't assign a face of dim 0 (point) a user_index...
            // Works in 2D and 3D, but not 1D
            // const unsigned int face_user_index = face->user_index();

            for (unsigned int idof_face=0; idof_face<dofs_per_face; ++idof_face) {
                // Get the cell dof index based on the current face dof index.
                // For example, there might be 9 dofs on the face.
                // The 5th dof of that face might correspond to the 24th dof on the cell
                unsigned int idof_cell = fe_system.face_to_cell_index(idof_face, iface);

                const dealii::types::global_dof_index global_idof_index = dof_indices[idof_cell];

                // If dof is not already in our hashmap, then insert it into our set of surface indices.
                // Even though the set is unique, we are really only creating a vector of indices and boundary ID,
                // and therefore can't just insert.
                // However, do not add if it is not locally owned, it will get picked up by another processor


                // For linear elasticity, need access to locally RELEVANT volume_nodes. Not just the locally owned ones.
                if ( locally_relevant_surface_dof_indices_set.find(global_idof_index) == locally_relevant_surface_dof_indices_set.end() ) {
                    locally_relevant_surface_dof_indices_set.insert(global_idof_index);
                    locally_relevant_surface_nodes_indices.push_back(global_idof_index);
                    locally_relevant_surface_nodes_boundary_id.push_back(boundary_id);
                    // Refer to the nonmenclature of
                    // https://www.dealii.org/current/doxygen/deal.II/classFiniteElement.html#ae2ea16b60a6fc644a9bc7097703a53e8
                    const unsigned int i = idof_cell;
                    const unsigned int component = fe_system.system_to_component_index(i).first;
                    const unsigned int shape_within_base = fe_system.system_to_component_index(i).second;
                    unsigned int point_index = 0;
                    if ( shape_within_base_found.find(shape_within_base) == shape_within_base_found.end() ) {
                        // If the point does not exist, add it to the list of points.
                        shape_within_base_found.insert(shape_within_base);

                        dealii::Point<dim> point;
                        point[component] = volume_nodes[global_idof_index];
                        locally_relevant_surface_points.push_back(point);
                        point_index = locally_relevant_surface_points.size()-1;
                        shape_function_within_base_to_point_index[shape_within_base] = point_index;
                    } else {
                        // The point has already been visited.
                        // Simply complete its Point data entries.
                        point_index = shape_function_within_base_to_point_index[shape_within_base];
                        locally_relevant_surface_points[point_index][component] = volume_nodes[global_idof_index];
                    }
                    //if (global_idof_index == 1682 || global_idof_index == 1683) {
                    //    std::cout << "Point 1682 or 1683 " << locally_relevant_surface_points[point_index] << std::endl;
                    //}

                    global_index_to_point_and_axis[global_idof_index] = std::make_pair(point_index, component);
                    point_and_axis_to_global_index[std::make_pair(point_index, component)] = global_idof_index;
                }

                if (locally_owned_dofs_grid.is_element(global_idof_index)) {
                    if ( locally_owned_surface_dof_indices_set.find(global_idof_index) == locally_owned_surface_dof_indices_set.end() ) {
                        locally_owned_surface_dof_indices_set.insert(global_idof_index);
                        locally_owned_surface_nodes_indices.push_back(global_idof_index);
                    }
                }
            }
        } // iface loop

    } // cell loop
    //std::cout << "I own " << locally_relevant_surface_nodes_indices.size() << " surface nodes" << std::endl;
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>

VectorType
HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::transform_surface_nodes(std::function<dealii::Point<dim>(dealii::Point<dim>)> transformation) const
{
    VectorType new_surface_nodes(surface_nodes);
    auto index = surface_to_volume_indices.begin();
    auto new_node = new_surface_nodes.begin();
    for (; index != surface_to_volume_indices.end(); ++index, ++new_node) {
        const dealii::types::global_dof_index global_idof_index = *index;
        //const std::pair<unsigned int, unsigned int> ipoint_component = global_index_to_point_and_axis[global_idof_index];
        //const auto ipoint_component = global_index_to_point_and_axis.at(global_idof_index);
        const std::pair<unsigned int, unsigned int> ipoint_component = global_index_to_point_and_axis.at(global_idof_index);
        const unsigned int ipoint = ipoint_component.first;
        const unsigned int component = ipoint_component.second;
        dealii::Point<dim> point;
        for (int d=0;d<dim;d++) {
            point[d] = locally_relevant_surface_points[ipoint][d];
        }
        dealii::Point<dim> new_point = transformation(point);
        *new_node = new_point[component];
    }
    new_surface_nodes.update_ghost_values();
    return new_surface_nodes;
}

template <int dim, typename real, typename MeshType, typename VectorType, typename DoFHandlerType>
void HighOrderGrid<dim,real,MeshType,VectorType,DoFHandlerType>::output_results_vtk (const unsigned int cycle) const
{
    std::string master_fn = "Mesh-" + dealii::Utilities::int_to_string(dim, 1) +"D_GridP"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
    master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
    pcout << "Outputting grid: " << master_fn << " ... " << std::endl;

    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler (dof_handler_grid);

    std::vector<std::string> solution_names;
    for(int d=0;d<dim;++d) {
        if (d==0) solution_names.push_back("x");
        if (d==1) solution_names.push_back("y");
        if (d==2) solution_names.push_back("z");
    }
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, dealii::DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector (volume_nodes, solution_names, dealii::DataOut<dim>::type_dof_data, data_component_interpretation);

    dealii::Vector<float> subdomain(triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain[i] = triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // const GridPostprocessor<dim> grid_post_processor;
    // data_out.add_data_vector (volume_nodes, grid_post_processor);

    VectorType jacobian_determinant;
    jacobian_determinant.reinit(locally_owned_dofs_grid, ghost_dofs_grid, mpi_communicator);
    const unsigned int n_dofs_per_cell = fe_system.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> dofs_indices(n_dofs_per_cell);
    const std::vector< dealii::Point<dim> > &points = fe_system.get_unit_support_points();
    std::vector<real> jac_det;
    for (auto cell = dof_handler_grid.begin_active(); cell!=dof_handler_grid.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        jac_det = evaluate_jacobian_at_points(volume_nodes, cell, points);
        cell->get_dof_indices (dofs_indices);
        for (unsigned int i=0; i<n_dofs_per_cell; ++i) {
            jacobian_determinant[dofs_indices[i]] = jac_det[i];
        }
    }

    jacobian_determinant.update_ghost_values();
    std::vector<std::string> jacobian_name;
    for (int d=0;d<dim;++d) {
        jacobian_name.push_back("JacobianDeterminant" + dealii::Utilities::int_to_string(d, 1));
    }
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> jacobian_component_interpretation(dim, dealii::DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector (jacobian_determinant, jacobian_name, dealii::DataOut<dim>::type_dof_data, jacobian_component_interpretation);

    typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::curved_boundary;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::no_curved_cells;

    const dealii::Mapping<dim> &mapping = (*(mapping_fe_field));
    const int n_subdivisions = max_degree;;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    data_out.build_patches(mapping, n_subdivisions, curved);
    const bool write_higher_order_cells = (dim>1) ? true : false;
    dealii::DataOutBase::VtkFlags vtkflags(0.0,cycle,true,dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::best_compression,write_higher_order_cells);
    data_out.set_flags(vtkflags);

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::string filename = "Mesh-" + dealii::Utilities::int_to_string(dim, 1) +"D_GridP"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
    filename += dealii::Utilities::int_to_string(cycle, 4) + "." + dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            std::string fn = "Mesh-" + dealii::Utilities::int_to_string(dim, 1) +"D_GridP"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + "." + dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }

}

template <int dim>
void GridPostprocessor<dim>::evaluate_vector_field (
    const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_solution_points = inputs.solution_values.size();
    Assert (computed_quantities.size() == n_solution_points, dealii::ExcInternalError());
    Assert (inputs.solution_values[0].size() == dim, dealii::ExcInternalError());
    std::vector<std::string> names = get_names ();
    for (unsigned int q=0; q<n_solution_points; ++q) {
        computed_quantities[q].grow_or_shrink(names.size());
    }
    for (unsigned int q=0; q<n_solution_points; ++q) {

        // const dealii::Vector<double>              &uh                = inputs.solution_values[q];
        const std::vector<dealii::Tensor<1,dim> > &duh               = inputs.solution_gradients[q];
        // const std::vector<dealii::Tensor<2,dim> > &dduh              = inputs.solution_hessians[q];
        // const dealii::Tensor<1,dim>               &normals           = inputs.normals[q];
        // const dealii::Point<dim>                  &evaluation_points = inputs.evaluation_points[q];

        unsigned int current_data_index = 0;

        dealii::Tensor<2,dim,double> jacobian;
        for (unsigned int a=0;a<dim;++a) {
            for (int d=0;d<dim;++d) {
                jacobian[a][d] = duh[a][d];
                std::cout << jacobian[a][d] << " ";
            }
        }
        computed_quantities[q](current_data_index++) = determinant(jacobian);
        std::cout << "Jac: " << computed_quantities[q](current_data_index) << std::endl;

        if (computed_quantities[q].size() != current_data_index) {
            std::cout << " Did not assign a value to all the data. Missing " << computed_quantities[q].size() - current_data_index << " variables."
                      << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                      << std::endl;
        }
    }

}

template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> GridPostprocessor<dim>
::get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation;
    interpretation.push_back (DCI::component_is_scalar); // Jacobian

    std::vector<std::string> names = get_names();
    if (names.size() != interpretation.size()) {
        std::cout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}


template <int dim>
std::vector<std::string> GridPostprocessor<dim>::get_names () const
{
    std::vector<std::string> names;
    names.push_back ("JacobianDeterminant");
    return names;
}

template <int dim>
dealii::UpdateFlags GridPostprocessor<dim>::get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values | dealii::update_gradients;
}


template class HighOrderGrid <PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class HighOrderGrid <PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class HighOrderGrid <PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
