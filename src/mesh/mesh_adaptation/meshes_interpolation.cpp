#include "meshes_interpolation.h"

#include "dg/dg_factory.hpp"
#include "mesh/high_order_grid.h"

#include <deal.II/base/function.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/grid/grid_in.h>

namespace PHiLiP {

    template <int dim, int nstate, typename MeshType>
    MeshInterpolation<dim, nstate, MeshType>::MeshInterpolation(std::ostream& out_stream)
        : out(out_stream)
    {}

    template <int dim, int nstate, typename MeshType>
    void MeshInterpolation<dim, nstate, MeshType>::perform_mesh_interpolation_test(
        const std::shared_ptr<DGBase<dim, double, MeshType>>& source_dg,
        const Parameters::AllParameters& param,
        const int poly_degree_interpolation,
        const std::string& target_mesh_file) const
    {
        out << "Mesh interpolation begin\n"
            << "Target Mesh: " << target_mesh_file << "\n"
            << "Polynomial degree: " << poly_degree_interpolation << "\n";

        // Output source solution
        out << "Outputting source solution\n";
        source_dg->output_results_vtk(88888);
        out << "Saved: solution-88888.vtu\n";

        // Create a new triangulation of the correct MeshType
        std::shared_ptr<MeshType> target_triangulation;

        // Use the same MPI communicator as the source
        if constexpr (std::is_same_v<MeshType, dealii::Triangulation<dim>>) {
            target_triangulation = std::make_shared<MeshType>();
        }
        else if constexpr (std::is_same_v<MeshType, dealii::parallel::shared::Triangulation<dim>>) {
            target_triangulation = std::make_shared<MeshType>(
                source_dg->triangulation->get_communicator());
        }
        else if constexpr (std::is_same_v<MeshType, dealii::parallel::distributed::Triangulation<dim>>) {
            target_triangulation = std::make_shared<MeshType>(
                source_dg->triangulation->get_communicator());
        }

        // Read the mesh file directly into this triangulation
        out << "Reading target mesh from: " << target_mesh_file << "\n";
        dealii::GridIn<dim> grid_in;
        grid_in.attach_triangulation(*target_triangulation);
        std::ifstream msh_file(target_mesh_file);
        if (!msh_file.good()) {
            out << "Error: Could not open mesh file " << target_mesh_file << "\n";
            return;
        }
        grid_in.read_msh(msh_file);

        // Create target DG with specified polynomial degree
        out << "Creating target DG object...\n";
        auto target_dg = DGFactory<dim, double, MeshType>::create_discontinuous_galerkin(
            &param,
            poly_degree_interpolation,      // poly_degree
            poly_degree_interpolation,      // poly_degree_max  
            1,                              // poly_degree_grid
            target_triangulation);

        target_dg->allocate_system();

        out << "Mesh Statistics (P=" << poly_degree_interpolation << "):\n"
            << " Source: " << source_dg->triangulation->n_active_cells()
            << " cells, " << source_dg->dof_handler.n_dofs() << " DoFs\n"
            << " Target: " << target_dg->triangulation->n_active_cells()
            << " cells, " << target_dg->dof_handler.n_dofs() << " DoFs\n";

        // Create a function that evaluates the source solution
        using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
        dealii::Functions::FEFieldFunction<dim, dealii::DoFHandler<dim>, DealiiVector>
            source_function(source_dg->dof_handler, source_dg->solution);

        const unsigned int map_degree = (poly_degree_interpolation == 0 ? 1 : poly_degree_interpolation);
        dealii::MappingQGeneric<dim> mapping(map_degree);

        // Perform node by node interpolation
        out << "Performing interpolation...\n";
        for (const auto& cell : target_dg->dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;

            const auto& fe = cell->get_fe();
            const unsigned int dofs_per_cell = fe.dofs_per_cell;
            std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
            const auto& unit_support_points = fe.get_unit_support_points();

            // For each DoF on this cell
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                dealii::Point<dim> unit_point;
                if (i < unit_support_points.size()) {
                    unit_point = unit_support_points[i];
                }
                else {
                    unit_point = unit_support_points[i % unit_support_points.size()];
                }

                // Map unit support points to real coordinates
                const dealii::Point<dim> real_point = mapping.transform_unit_to_real_cell(cell, unit_point);

                try {
                    unsigned int component = 0;
                    if (fe.n_components() > 1) {
                        auto comp_idx = fe.system_to_component_index(i);
                        component = comp_idx.first;
                    }
                    double value = source_function.value(real_point, component);
                    target_dg->solution(local_dof_indices[i]) = value;
                }
                catch (const std::exception& e) {
                    // Point not found in source mesh or other error
                    out << "Warning: Interpolation failed at point " << real_point
                        << ", setting to 0.0\n";
                    target_dg->solution(local_dof_indices[i]) = 0.0;
                }
            }
        }

        target_dg->solution.compress(dealii::VectorOperation::insert);
        target_dg->solution.update_ghost_values();

        // Output target solution
        out << "Outputting target solution\n";
        target_dg->output_results_vtk(77777);
        out << "Saved: solution-77777.vtu\n";

        out << "Mesh interpolation test completed\n";
    }


    // 2D instantiations
    template class MeshInterpolation<2, 1, dealii::Triangulation<2>>;
    template class MeshInterpolation<2, 2, dealii::Triangulation<2>>;
    template class MeshInterpolation<2, 3, dealii::Triangulation<2>>;
    template class MeshInterpolation<2, 4, dealii::Triangulation<2>>;
    template class MeshInterpolation<2, 5, dealii::Triangulation<2>>;

    template class MeshInterpolation<2, 1, dealii::parallel::shared::Triangulation<2>>;
    template class MeshInterpolation<2, 2, dealii::parallel::shared::Triangulation<2>>;
    template class MeshInterpolation<2, 3, dealii::parallel::shared::Triangulation<2>>;
    template class MeshInterpolation<2, 4, dealii::parallel::shared::Triangulation<2>>;
    template class MeshInterpolation<2, 5, dealii::parallel::shared::Triangulation<2>>;

#if PHILIP_DIM != 1
    template class MeshInterpolation<2, 1, dealii::parallel::distributed::Triangulation<2>>;
    template class MeshInterpolation<2, 2, dealii::parallel::distributed::Triangulation<2>>;
    template class MeshInterpolation<2, 3, dealii::parallel::distributed::Triangulation<2>>;
    template class MeshInterpolation<2, 4, dealii::parallel::distributed::Triangulation<2>>;
    template class MeshInterpolation<2, 5, dealii::parallel::distributed::Triangulation<2>>;
#endif

} // namespace PHiLiP