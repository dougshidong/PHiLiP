#include "meshes_interpolation.h"

#include "dg/dg_factory.hpp"
#include "mesh/high_order_grid.h"

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <cstddef> // For offsetof

#if defined(DEAL_II_WITH_MPI)
#  include <mpi.h>
#endif

/**
 * NOTE �C Current Limitations
 * ---------------------------
 * - MPI support is implemented, but performance is compromised due to remote evaluations.
 * - Out-of-domain DoFs: nodes that fall outside the source mesh are set to zero.
 * - Dimensionality: code has been tested in 2D; 3D grids are still untested.
 *
 * TODO
 * ----
 * - Optimize parallel support.
 * - Implement a smarter method for handling out-of-domain evaluations.
 * - Extend and validate the algorithm for 3D meshes.
 */

namespace PHiLiP
{   // Constructor
    template <int dim, int nstate, typename MeshType>
    MeshInterpolation<dim, nstate, MeshType>::MeshInterpolation(
        std::ostream& out_stream)
        : out(out_stream)
    {
    }
    // mesh_interpolation_test
    template <int dim, int nstate, typename MeshType>
    std::shared_ptr<DGBase<dim, double, MeshType>> MeshInterpolation<dim, nstate, MeshType>::perform_mesh_interpolation(
            const std::shared_ptr<DGBase<dim, double, MeshType>>& source_dg,
            const Parameters::AllParameters& param,
            const int poly_degree_interpolation,
            const std::string& target_mesh_file) const
    {
        out << "Mesh interpolation begin\n"
            << "Target Mesh: " << target_mesh_file << "\n"
            << "Polynomial degree: " << poly_degree_interpolation << "\n";

        // Output source solution, this is not working with the unit test since there is no writing premission.
        //out << "Outputting source solution\n";
        //source_dg->output_results_vtk(00000);
        //out << "Saved: solution-00000.vtu\n";

        // SETUP TARGET MESH
        std::shared_ptr<MeshType> target_triangulation;
        if constexpr (std::is_same_v<MeshType, dealii::Triangulation<dim>>)
        {
            target_triangulation = std::make_shared<MeshType>();
        }
        else 
        {
            target_triangulation = std::make_shared<MeshType>(
                source_dg->triangulation->get_communicator());
        }

        out << "Reading target mesh from: " << target_mesh_file << "\n";
        dealii::GridIn<dim> grid_in;
        grid_in.attach_triangulation(*target_triangulation);
        std::ifstream msh_file(target_mesh_file);
        if (!msh_file.good())
        {
            out << "Error: Could not open mesh file " << target_mesh_file << "\n";
            return nullptr;
        }
        grid_in.read_msh(msh_file);

        // CREATE DG OBJECT FOR THE TARGET MESH
        out << "Creating target DG object...\n";
        auto target_dg =
            DGFactory<dim, double, MeshType>::create_discontinuous_galerkin(
                &param,
                poly_degree_interpolation,
                poly_degree_interpolation + 1,
                1,
                target_triangulation);
        target_dg->allocate_system();

        out << "Mesh Statistics (P=" << poly_degree_interpolation << "):\n"
            << " Source: " << source_dg->triangulation->n_active_cells()
            << " cells, " << source_dg->dof_handler.n_dofs() << " DoFs\n"
            << " Target: " << target_dg->triangulation->n_active_cells()
            << " cells, " << target_dg->dof_handler.n_dofs() << " DoFs\n";

        // CREATE THE EVALUATION FUNCTION
        using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
        dealii::Functions::FEFieldFunction<dim, dealii::DoFHandler<dim>, DealiiVector>
            source_function(source_dg->dof_handler, source_dg->solution);

        // CREATE THE MAPPING FUNCTION
        const unsigned int map_degree = (poly_degree_interpolation == 0 ? 1 : poly_degree_interpolation);
        dealii::MappingQGeneric<dim> mapping(map_degree);

        // LOCAL EVALUATION
        std::vector<RemotePointQuery<dim>> local_queries;
        out << "Attempting local evaluation...\n";

        for (const auto& cell : target_dg->dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            const auto& fe = cell->get_fe();
            const unsigned int dofs_per_cell = fe.dofs_per_cell;
            // Obtain dof indices for the current cell
            std::vector<dealii::types::global_dof_index> local_dof_indices(
                dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
            // Obtain the local points for the current cell
            const auto& unit_support_points = fe.get_unit_support_points();

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                dealii::Point<dim> unit_point =
                    (i < unit_support_points.size()) ?
                    unit_support_points[i] :
                    unit_support_points[i % unit_support_points.size()];

                // Transform the local point to the point in physical space
                const dealii::Point<dim> real_point =
                    mapping.transform_unit_to_real_cell(cell, unit_point);

                unsigned int component = 0;
                if (fe.n_components() > 1)
                    component = fe.system_to_component_index(i).first;
                try
                {
                // Evaluate the solution at current location for specific component
                    const double value =
                        source_function.value(real_point, component);
                    target_dg->solution(local_dof_indices[i]) = value;
                }
                catch (const std::exception& e)
                {
                    // This point is on another process; add it to the query list.
                    local_queries.push_back(
                        { real_point, local_dof_indices[i], component });
                }
            }
        }

        // FOR MPI ONLY
        // EXCHANGE LOCAL QUERIES FROM EACH RANKS
        if constexpr (!std::is_same_v<MeshType, dealii::Triangulation<dim>>)
        {
#if defined(DEAL_II_WITH_MPI)
            // Obtain the communicator
            MPI_Comm communicator = source_dg->triangulation->get_communicator();
            int n_procs;
            MPI_Comm_size(communicator, &n_procs);

            out << "Exchanging " << local_queries.size()
                << " local queries across " << n_procs << " processes...\n";

            // Create a vector stores the queries number for each rank
            std::vector<int> query_counts(n_procs);
            int local_query_count = local_queries.size();
            MPI_Allgather(&local_query_count, 1, MPI_INT, query_counts.data(), 1, MPI_INT, communicator);

            std::vector<RemotePointQuery<dim>> global_queries;
            // Obtain the total number of queries
            int total_queries = 0;
            // Stores the starting index in global queries where rank i data should go
            std::vector<int> displacements(n_procs);
            for (int i = 0; i < n_procs; ++i)
            {
                displacements[i] = total_queries;
                total_queries += query_counts[i];
            }

            // Resize the global queries
            global_queries.resize(total_queries);
            MPI_Datatype mpi_query_type;                    
            const int n_query_items = 3;                      

            int blocklengths_query[] = { dim, 1, 1 };

            // Compute the offest bytes
            MPI_Aint mpi_displacements_query[] = {
              offsetof(RemotePointQuery<dim>, physical_point),
              offsetof(RemotePointQuery<dim>, target_dof_index),
              offsetof(RemotePointQuery<dim>, component)
            };

            // MPI type of each field, in the same order as above
            MPI_Datatype types_query[] = {
              MPI_DOUBLE,             // Coordinates
              MPI_UNSIGNED_LONG_LONG, // DoF index
              MPI_UNSIGNED            // Component number
            };

            // Create and commit the derived datatype 
            MPI_Type_create_struct(
                n_query_items,
                blocklengths_query,
                mpi_displacements_query,
                types_query,
                &mpi_query_type);
            MPI_Type_commit(&mpi_query_type);

            MPI_Allgatherv(
                local_queries.data(),              // send buffer
                local_queries.size(),              // send count
                mpi_query_type,                    // send datatype

                global_queries.data(),             // receive buffer
                query_counts.data(),               // receive counts
                displacements.data(),              // receive displacement
                mpi_query_type,                    // receive datatype

                communicator);                     // MPI communicator

            // Done with the custom type; free the handle 
            MPI_Type_free(&mpi_query_type);

            // REMOTE EVALUATION
            out << "Performing remote evaluation on " << total_queries << " total points...\n";
            std::vector<RemotePointResult<dim>> local_results;
            for (const auto& query : global_queries)
            {
                try
                {
                    const double value =
                        source_function.value(query.physical_point, query.component);
                    local_results.push_back({ query.target_dof_index, value });
                }
                catch (const std::exception& e)
                {
                    // This point belongs to other proccessor. Ignore.
                }
            }

            // EXCHANGE GLOBAL RESULTS
            out << " Exchanging " << local_results.size() << " found results...\n";
            std::vector<int> result_counts(n_procs);
            int local_result_count = local_results.size();
            // Stores numbers of result from each ranks 
            MPI_Allgather(&local_result_count, 1, MPI_INT, result_counts.data(), 1, MPI_INT, communicator);

            // Global result
            std::vector<RemotePointResult<dim>> global_results;
            int total_results = 0;
            for (int i = 0; i < n_procs; ++i)
            {
                // Stores the starting index in global queries where rank i data should go
                displacements[i] = total_results;
                total_results += result_counts[i];
            }
            global_results.resize(total_results);

            MPI_Datatype mpi_result_type;
            const int n_result_items = 2;
            int blocklengths_result[] = { 1, 1 };
            // Calculate the bytes offset to each types of result
            MPI_Aint mpi_displacements_result[] = {
              offsetof(RemotePointResult<dim>, target_dof_index),
              offsetof(RemotePointResult<dim>, value) };
            MPI_Datatype result_types[] = { MPI_UNSIGNED_LONG_LONG, MPI_DOUBLE };
            MPI_Type_create_struct(n_result_items, blocklengths_result, mpi_displacements_result, result_types, &mpi_result_type);
            MPI_Type_commit(&mpi_result_type);

            MPI_Allgatherv(
                local_results.data(), // send buffer
                local_results.size(), // send counts
                mpi_result_type,      // send data type

                global_results.data(), // receive buffer
                result_counts.data(),  // receive counts
                displacements.data(),  // receive displacements
                mpi_result_type,       // receive data type

                communicator);
            MPI_Type_free(&mpi_result_type); //clean up the created struct

            // FINAL SOLUTION UPDATE
            out << " Updating solution with " << total_results << " remote values...\n";
            for (const auto& result : global_results)
            {
                // Check if this process owns the DoF before writing.
                if (target_dg->dof_handler.locally_owned_dofs().is_element(result.target_dof_index))
                {
                    target_dg->solution(result.target_dof_index) = result.value;
                }
            }
#endif
        }

        // For serial runs, points in local_queries are outside the domain. Set to 0.
        if constexpr (std::is_same_v<MeshType, dealii::Triangulation<dim>>)
        {
            if (!local_queries.empty())
                out << "Warning: " << local_queries.size() << " points were outside the source domain and set to 0.0\n";
            for (const auto& query : local_queries)
            {
                target_dg->solution(query.target_dof_index) = 0.0;
            }
        }

        // Sends out the insetion for each rank 
        target_dg->solution.compress(dealii::VectorOperation::insert);
        // Pull in everyones else's update
        target_dg->solution.update_ghost_values();
       
        //Output target solution, not working with the unit test
        out << "Outputting target solution\n";
        target_dg->output_results_vtk(11111);
        out << "Saved: solution-11111.vtu\n";

        out << "Mesh interpolation test completed\n";

        return target_dg;
    }

    // 2D instantiations
#if PHILIP_DIM == 2
    template class MeshInterpolation<2, 1, dealii::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 2, dealii::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 3, dealii::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 4, dealii::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 5, dealii::Triangulation<PHILIP_DIM>>;

    template class MeshInterpolation<2, 1, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 2, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 3, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 4, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 5, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

    template class MeshInterpolation<2, 1, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 2, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 3, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 4, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
    template class MeshInterpolation<2, 5, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace PHiLiP