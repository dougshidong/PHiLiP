#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/solver_cg.h>
//#include <deal.II/lac/precondition.h>
//#include <deal.II/lac/precondition_block.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "meshmover_linear_elasticity.hpp"

namespace PHiLiP {
namespace MeshMover {

    // template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    // LinearElasticity<dim,real,VectorType,DoFHandlerType>::LinearElasticity(
    //     const HighOrderGrid<dim,real,VectorType,DoFHandlerType> &high_order_grid,
    //     const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector)
    //   : triangulation(*(high_order_grid.triangulation))
    //   , mapping_fe_field(high_order_grid.mapping_fe_field)
    //   , dof_handler(high_order_grid.dof_handler_grid)
    //   , quadrature_formula(dof_handler.get_fe().degree + 1)
    //   , mpi_communicator(MPI_COMM_WORLD)
    //   , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
    //   , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
    //   , pcout(std::cout, this_mpi_process == 0)
    //   , boundary_ids_vector(high_order_grid.surface_to_volume_indices)
    //   , boundary_displacements_vector(boundary_displacements_vector)
    // { 
    //     AssertDimension(boundary_displacements_vector.size(), boundary_ids_vector.size());
    // }

    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    LinearElasticity<dim,real,VectorType,DoFHandlerType>::LinearElasticity(
        const HighOrderGrid<dim,real,VectorType,DoFHandlerType> &high_order_grid,
        const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector)
      : LinearElasticity<dim,real,VectorType,DoFHandlerType> (
          *(high_order_grid.triangulation),
          high_order_grid.mapping_fe_field,
          high_order_grid.dof_handler_grid,
          high_order_grid.surface_to_volume_indices,
          boundary_displacements_vector)
    { }

    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    LinearElasticity<dim,real,VectorType,DoFHandlerType>::LinearElasticity(
        const Triangulation &_triangulation,
        const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> mapping_fe_field,
        const DoFHandlerType &_dof_handler,
        const dealii::LinearAlgebra::distributed::Vector<int> &_boundary_ids_vector,
        const dealii::LinearAlgebra::distributed::Vector<double> &_boundary_displacements_vector)
      : triangulation(_triangulation)
      , mapping_fe_field(mapping_fe_field)
      , dof_handler(_dof_handler)
      , quadrature_formula(dof_handler.get_fe().degree + 1)
      , mpi_communicator(MPI_COMM_WORLD)
      , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
      , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , pcout(std::cout, this_mpi_process == 0)
      , boundary_ids_vector(_boundary_ids_vector)
      , boundary_displacements_vector(_boundary_displacements_vector)
    { 
        AssertDimension(boundary_displacements_vector.size(), boundary_ids_vector.size());

        boundary_displacements_vector.update_ghost_values();
        setup_system();
    }

    // template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    // LinearElasticity<dim,real,VectorType,DoFHandlerType>::LinearElasticity(
    //     const HighOrderGrid<dim,real,VectorType,DoFHandlerType> &high_order_grid,
    //     const std::vector<dealii::Tensor<1,dim,real>> &boundary_displacements_tensors)
    //   : LinearElasticity(high_order_grid, boundary_displacements_vector(tensor_to_vector(boundary_displacements_tensors))
    // { }

    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    dealii::LinearAlgebra::distributed::Vector<double> LinearElasticity<dim,real,VectorType,DoFHandlerType>::
    tensor_to_vector(const std::vector<dealii::Tensor<1,dim,real>> &boundary_displacements_tensors) const
    {
        (void) boundary_displacements_tensors;
        dealii::LinearAlgebra::distributed::Vector<double> boundary_displacements_vector;
        return boundary_displacements_vector;
    }

    //template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    //LinearElasticity<dim,real,VectorType,DoFHandlerType>::~LinearElasticity() { dof_handler.clear(); }

    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    VectorType LinearElasticity<dim,real,VectorType,DoFHandlerType>::get_volume_displacements()
    {
        pcout << std::endl << "Solving linear elasticity problem for volume displacements..." << std::endl;
        solve_timestep();
        // displacement_solution = 0;
        // all_constraints.distribute(displacement_solution);
        displacement_solution.update_ghost_values();
        return displacement_solution;
    }
    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    void LinearElasticity<dim,real,VectorType,DoFHandlerType>::setup_system()
    {
        //dof_handler.distribute_dofs(fe_system);
        //dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
        dealii::IndexSet ghost_dofs = locally_relevant_dofs;
        ghost_dofs.subtract_set(locally_owned_dofs);
        //ghost_dofs.print(std::cout);

        system_rhs.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
        system_rhs_unconstrained.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
        displacement_solution.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);

        // Set the hanging node constraints
        all_constraints.clear();
        all_constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, all_constraints);

        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);

        // Set the Dirichlet BC constraint
        // Only add Dirichlet BC when there is currently no constraint.
        // Note that we set ALL surfaces to be an inhomogeneous BC.
        // In a way, it allows us to detect surface DoFs in the matrix by checking whether there is an inhomogeneous constraight.
        // If there is another constraint, it's because it's a hanging node.
        // In that case, we choose to satisfy the regularity properties of the element and ignoring the Dirichlet BC.
        const auto &partitionner = boundary_ids_vector.get_partitioner();
        for (unsigned int isurf = 0; isurf < boundary_ids_vector.size(); ++isurf) {
            const bool is_accessible = partitionner->in_local_range(isurf) || partitionner->is_ghost_entry(isurf);
            if (is_accessible) {
                const unsigned int iglobal_row = boundary_ids_vector[isurf];
                const double dirichlet_value = boundary_displacements_vector[isurf];
                Assert(all_constraints.can_store_line(iglobal_row), dealii::ExcInternalError());
                all_constraints.add_line(iglobal_row);
                all_constraints.set_inhomogeneity(iglobal_row, dirichlet_value+1e-15);
            }
        }
        all_constraints.close();
        dealii::IndexSet temp_locally_active_dofs;
        dealii::DoFTools::extract_locally_active_dofs(dof_handler, temp_locally_active_dofs);
        const std::vector<dealii::IndexSet> local_dofs_per_process = dealii::Utilities::MPI::all_gather(mpi_communicator, dof_handler.locally_owned_dofs());
        AssertThrow(all_constraints.is_consistent_in_parallel(local_dofs_per_process,
                    temp_locally_active_dofs,
                    mpi_communicator,
                    /*verbose*/ true),
                    dealii::ExcInternalError());


        dealii::DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                        sparsity_pattern,
                                        all_constraints,
                                        /*keep constrained dofs*/ true);
        dealii::SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                                   dof_handler.locally_owned_dofs(),
                                                   mpi_communicator,
                                                   locally_relevant_dofs);
        system_matrix.reinit(locally_owned_dofs,
                             locally_owned_dofs,
                             sparsity_pattern,
                             mpi_communicator);
        system_matrix_unconstrained.reinit(locally_owned_dofs,
                                 locally_owned_dofs,
                                 sparsity_pattern,
                                 mpi_communicator);

        pcout << "    Number of active cells: " << triangulation.n_active_cells() << std::endl;
        pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
    }
    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    void LinearElasticity<dim,real,VectorType,DoFHandlerType>::assemble_system()
    {
        pcout << "    Assembling MeshMover::LinearElasticity system..." << std::flush;

        setup_system();

        system_rhs    = 0;
        system_matrix = 0;
        system_rhs_unconstrained    = 0;
        system_matrix_unconstrained = 0;
        const dealii::FESystem<dim> &fe_system = dof_handler.get_fe(0);
        dealii::FEValues<dim> fe_values(
            *mapping_fe_field,
            fe_system,
            quadrature_formula,
            dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);
        const unsigned int dofs_per_cell = fe_system.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        dealii::Vector<double>     cell_rhs(dofs_per_cell);

        std::vector<double> youngs_modulus(n_q_points, 1.0);
        std::vector<double> poissons_ratio(n_q_points, 0.4);

        std::vector<double> lame_lambda_values(n_q_points);
        std::vector<double> lame_mu_values(n_q_points);
        for (unsigned int iquad = 0; iquad < n_q_points; ++iquad) {
            const double E = youngs_modulus[iquad];
            const double nu = poissons_ratio[iquad];
            lame_lambda_values[iquad] = E*nu/((1.0+nu)*(1-2.0*nu));
            lame_mu_values[iquad] = 0.5*E/(1.0+nu);
        }

        dealii::Functions::ZeroFunction<dim> body_force(dim);
        std::vector<dealii::Vector<double>> body_force_values(n_q_points,dealii::Vector<double>(dim));

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;

            cell_matrix = 0;
            cell_rhs    = 0;
            fe_values.reinit(cell);

            //lame_lambda.value_list(fe_values.get_quadrature_points(), lame_lambda_values);
            //lame_mu.value_list(fe_values.get_quadrature_points(), lame_mu_values);

            for (unsigned int itest = 0; itest < dofs_per_cell; ++itest) {

                const unsigned int component_test = fe_system.system_to_component_index(itest).first;

                for (unsigned int itrial = 0; itrial < dofs_per_cell; ++itrial) {

                    const unsigned int component_trial = fe_system.system_to_component_index(itrial).first;

                    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                        //const SymmetricTensor<2, dim> grad_basis_i_grad_u
                        //cell_matrix(itest, itrial) += 
                        double value = lame_lambda_values[q_point]
                            * fe_values.shape_grad_component(itest, q_point, component_test)
                            * fe_values.shape_grad_component(itrial, q_point, component_trial);
                        value += lame_mu_values[q_point]
                            * fe_values.shape_grad_component(itest, q_point, component_trial)
                            * fe_values.shape_grad_component(itrial, q_point, component_test);
                        if (component_test == component_trial) {
                            value += lame_mu_values[q_point]
                                * fe_values.shape_grad_component(itest, q_point, component_test)
                                * fe_values.shape_grad_component(itrial, q_point, component_trial);
                        }
                        value *= fe_values.JxW(q_point);

                        cell_matrix(itest, itrial) += value;
                    }
                }
            }
            body_force.vector_value_list(fe_values.get_quadrature_points(), body_force_values);
            for (unsigned int itest = 0; itest < dofs_per_cell; ++itest) {
                const unsigned int component_test = fe_system.system_to_component_index(itest).first;
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                    cell_rhs(itest) += body_force_values[q_point][component_test] * fe_values.shape_value(itest, q_point);
                    cell_rhs(itest) *= fe_values.JxW(q_point);
                }
            }
            cell->get_dof_indices(local_dof_indices);
            //all_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
            const bool use_inhomogeneities_for_rhs = false;
            all_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs, use_inhomogeneities_for_rhs);
         // Unconstrained system matrix and right-hand side
         // std::cout << std::endl;
         // for (auto const &value: local_dof_indices) {
         //    std::cout << value << std::endl;
         // }
         // system_matrix.print(std::cout);
         // system_matrix_unconstrained.print(std::cout);
         //  cell_matrix.print(std::cout);
         //  system_matrix_unconstrained.add(local_dof_indices, cell_matrix);
         //  //system_matrix_unconstrained.add(local_dof_indices, local_dof_indices, cell_matrix);
         //  system_rhs_unconstrained.add(local_dof_indices, cell_rhs);
            //  // for (unsigned int itest = 0; itest < dofs_per_cell; ++itest) {
         //  //    std::vector<double> row_values(dofs_per_cell);
            //  //     for (unsigned int itrial = 0; itrial < dofs_per_cell; ++itrial) {
         //  //       row_values[itrial] = cell_matrix[itest][itrial];
         //  //    }
         //  //    system_rhs_unconstrained.add(local_dof_indices[itest], cell_rhs);
         //  // }
            //  //  for (unsigned int itest = 0; itest < dofs_per_cell; ++itest) {
            //  //      for (unsigned int itrial = 0; itrial < dofs_per_cell; ++itrial) {
         //  //        system_matrix_unconstrained.add(local_dof_indices, local_dof_indices, cell_matrix);
         //  //     }
         //  //     system_rhs_unconstrained.add(local_dof_indices, cell_rhs);
         //  //  }
            for (unsigned int itest = 0; itest < dofs_per_cell; ++itest) {
                for (unsigned int itrial = 0; itrial < dofs_per_cell; ++itrial) {
               system_matrix_unconstrained.add(local_dof_indices[itest], local_dof_indices[itrial], cell_matrix(itest,itrial));
            }
         }
         system_rhs_unconstrained.add(local_dof_indices, cell_rhs);

        } // active cell loop
        system_matrix.compress(dealii::VectorOperation::add);
        system_rhs.compress(dealii::VectorOperation::add);
        system_matrix_unconstrained.compress(dealii::VectorOperation::add);
        system_rhs_unconstrained.compress(dealii::VectorOperation::add);
    }
    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    void LinearElasticity<dim,real,VectorType,DoFHandlerType>::solve_timestep()
    {
        assemble_system();
        pcout << " norm of rhs is " << system_rhs.l2_norm() << std::endl;
        const unsigned int n_iterations = solve_linear_problem();
        pcout << "    Solver converged in " << n_iterations << " iterations." << std::endl;
    }

    // template <typename dealii::LinearAlgebra::distributed::Vector<double>>
    // template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    // dealii::LinearAlgebra::distributed::Vector<double>
    // LinearElasticity<dim,real,VectorType,DoFHandlerType>::
    // apply_dXvdXs_transpose(dealii::LinearAlgebra::distributed::Vector<double> &input_vector)
    // {
    //     std::vector<dealii::LinearAlgebra::distributed::Vector<double>> list_of_in_vector;
    //     list_of_in_vector.push_back(input_vector);
    //     std::vector<dealii::LinearAlgebra::distributed::Vector<double>> list_of_out_vector = apply_dXvdXs_transpose(list_of_in_vector);
    //     return list_of_in_vector[0];
    // }

    // template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    // std::vector<dealii::LinearAlgebra::distributed::Vector<double>>
    // LinearElasticity<dim,real,VectorType,DoFHandlerType>::
    // apply_dXvdXs_transpose(std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &list_of_vectors)
    // {
    //     const unsigned int n_rows = dof_handler.n_dofs();
    //     const unsigned int n_cols = list_of_vectors.size();
    //     const unsigned int max_per_row = n_cols;

    //     const dealii::SparsityPattern full_dsp(n_rows, n_cols, max_per_row);
    //     const dealii::IndexSet &row_part = dof_handler.locally_owned_dofs();
    //     dealii::IndexSet col_part(n_cols);
    //     col_part.add_range(0,n_cols);

    //     dealii::LinearAlgebra::distributed::Vector<double> trilinos_solution(system_rhs);
    //     all_constraints.set_zero(trilinos_solution);

    //     dealii::SolverControl solver_control(5000, 1e-12 * system_rhs.l2_norm());
    //     dealii::SolverCG<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi      precondition;
    //     precondition.initialize(system_matrix_unconstrained);
    //     //precondition.initialize(system_matrix);
    //     //solver.solve(system_matrix, trilinos_solution, system_rhs, precondition);

    //     //all_constraints.distribute(trilinos_solution);

    //     /// The use of the constrained linear operator is heavily discussed in:
    //     /// https://www.dealii.org/current/doxygen/deal.II/group__constraints.html
    //     //using trilinos_vector_type = VectorType;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_at = dealii::transpose_operator(dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained));
    //     const auto op_atmod = dealii::constrained_linear_operator(all_constraints, op_at);
    //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_at);
    //     const auto Ct   = transpose_operator(C);
    //     const auto Id_c = project_to_constrained_linear_operator(all_constraints, op_at);

    //     const unsigned int n_dirichlet_constraints = boundary_displacements_vector.size();
    //     const unsigned int n_surface_nodes = n_dirichlet_constraints;
    //     dXvdXs.clear();
    //     // const dealii::SparsityPattern full_surface_sparsity_pattern(n_rows, n_surface_nodes, n_surface_nodes);
    //     // dXvdXs.reinit(row_part, n_surface_nodes, full_surface_sparsity_pattern, mpi_communicator);
    //     pcout << "Solving for dXvdXs with " << n_dirichlet_constraints << "surface nodes..." << std::endl;

    //     unsigned int col = 0;
    //     for (auto &rhs_vector: list_of_vectors) {

    //         // Build RHS.
    //         VectorType Ctrhs;
    //         Ctrhs.reinit(rhs_vector);
    //         Ctrhs = Ct*rhs_vector;

    //         // Solution.
    //         VectorType op_inv_Ctrhs;

    //         // Solve modified system.
    //         dealii::deallog.depth_console(0);
    //         solver.solve(op_atmod, op_inv_Ctrhs, rhs_vector, precondition);

    //         //pcout << "Surface Dirichlet constraint " << iconstraint+1 << " out of " << n_dirichlet_constraints
    //         //      << " DoF constrained: " << iconstraint
    //         //      << "    Solver converged in " << solver_control.last_step() << " iterations." << std::endl;

    //         // Apply boundary condition
    //         VectorType dXvdXs_transpose_output;
    //         dXvdXs_transpose_output.reinit(rhs_vector);
    //         dXvdXs_transpose_output = op_at*C*op_inv_Ctrhs;
    //         dXvdXs_transpose_output *= -1.0;
    //         dXvdXs_transpose_output.add(rhs_vector);

    //         dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
    //         rw_vector.reinit(dXvdXs_transpose_output);

    //         dealii::LinearAlgebra::distributed::Vector<double> dXvdXs_i;
    //         dXvdXs_i.reinit(displacement_solution);
    //         dXvdXs_i.import(rw_vector, dealii::VectorOperation::insert);
    //         dXvdXs.push_back(dXvdXs_i);

    //         for (const auto &row: dof_handler.locally_owned_dofs()) {
    //             dXvdVector.set(row, col, dXvdXs_i[row]);
    //         }
    //     }
    //     dXvdVector.compress(dealii::VectorOperation::insert);
    // }

    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    void
    LinearElasticity<dim,real,VectorType,DoFHandlerType>
    ::apply_dXvdXs(
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &list_of_vectors,
        dealii::TrilinosWrappers::SparseMatrix &output_matrix)
    {
        assemble_system();

        const unsigned int n_rows = dof_handler.n_dofs();
        const unsigned int n_cols = list_of_vectors.size();
        //const unsigned int max_per_row = n_cols;

        const dealii::IndexSet &row_part = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        dealii::DynamicSparsityPattern full_dsp(n_rows, n_cols, row_part);
        for (const auto &i_row: row_part) {
            for (unsigned int i_col = 0; i_col < n_cols; ++i_col) {
                full_dsp.add(i_row, i_col);
            }
        }
        dealii::SparsityTools::distribute_sparsity_pattern(full_dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

        dealii::SparsityPattern full_sp;
        full_sp.copy_from(full_dsp);

        //const std::vector<dealii::IndexSet> col_parts = dealii::Utilities::MPI::create_evenly_distributed_partitioning(mpi_communicator, n_cols);
        //const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
        //const dealii::IndexSet &col_part = col_parts[this_mpi_process];
        const dealii::IndexSet col_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_cols);

        output_matrix.reinit(row_part, col_part, full_sp, mpi_communicator);

        dealii::SolverControl solver_control(5000, 1e-12 * system_rhs.l2_norm());
        dealii::SolverCG<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);
        dealii::TrilinosWrappers::PreconditionJacobi      precondition;
        precondition.initialize(system_matrix_unconstrained);
        //precondition.initialize(system_matrix);
        //solver.solve(system_matrix, trilinos_solution, system_rhs, precondition);

        //all_constraints.distribute(trilinos_solution);

        /// The use of the constrained linear operator is heavily discussed in:
        /// https://www.dealii.org/current/doxygen/deal.II/group__constraints.html
        using trilinos_vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
        using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
        const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
        const auto op_amod = dealii::constrained_linear_operator(all_constraints, op_a);
        const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
        const auto Ct   = transpose_operator(C);
        const auto Id_c = project_to_constrained_linear_operator(all_constraints, op_a);

        const unsigned int n_dirichlet_constraints = boundary_displacements_vector.size();
        dXvdXs.clear();
        pcout << "Solving for dXvdXs with " << n_dirichlet_constraints << "surface nodes..." << std::endl;

        unsigned int col = 0;
        for (auto &rhs_vector: list_of_vectors) {

            // Build RHS.
            dealii::LinearAlgebra::distributed::Vector<double> CtArhs;
            CtArhs.reinit(rhs_vector);
            CtArhs = Ct*op_a*rhs_vector;

            // Solution.
            dealii::LinearAlgebra::distributed::Vector<double> op_inv_CtArhs(rhs_vector);
            all_constraints.set_zero(op_inv_CtArhs);

            // Solve modified system.
            dealii::deallog.depth_console(0);
            solver.solve(op_amod, op_inv_CtArhs, CtArhs, precondition);

            // Apply boundary condition
            dealii::LinearAlgebra::distributed::Vector<double> dXvdXs_i_trilinos;
            dXvdXs_i_trilinos.reinit(CtArhs);
            dXvdXs_i_trilinos = C*op_inv_CtArhs;
            dXvdXs_i_trilinos *= -1.0;
            dXvdXs_i_trilinos += rhs_vector;

            dXvdXs.push_back(dXvdXs_i_trilinos);

            for (const auto &row: dof_handler.locally_owned_dofs()) {
                output_matrix.set(row, col, dXvdXs[col][row]);
            }
            col++;
        }
        output_matrix.compress(dealii::VectorOperation::insert);

    }

    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    void LinearElasticity<dim,real,VectorType,DoFHandlerType>::evaluate_dXvdXs()
    {
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> unit_rhs_vector;
        const unsigned int n_dirichlet_constraints = boundary_displacements_vector.size();
        dXvdXs.clear();
        pcout << "Solving for dXvdXs with " << n_dirichlet_constraints << "surface nodes..." << std::endl;
        for (unsigned int iconstraint = 0; iconstraint < n_dirichlet_constraints; iconstraint++) {

            dealii::LinearAlgebra::distributed::Vector<double> unit_rhs;
            unit_rhs.reinit(system_rhs);

            if (boundary_ids_vector.locally_owned_elements().is_element(iconstraint)) {
                const unsigned int constrained_row = boundary_ids_vector[iconstraint];
                unit_rhs[constrained_row] = 1.0;
            }
            unit_rhs.update_ghost_values();

            unit_rhs_vector.push_back(unit_rhs);
        }
        dealii::TrilinosWrappers::SparseMatrix dXvdXs_matrix;
        apply_dXvdXs(unit_rhs_vector, dXvdXs_matrix);
    }

    // template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    // void LinearElasticity<dim,real,VectorType,DoFHandlerType>::evaluate_dXvdXs()
    // {
    //     VectorType trilinos_solution(system_rhs);

    //     all_constraints.set_zero(trilinos_solution);

    //     dealii::SolverControl solver_control(5000, 1e-12 * system_rhs.l2_norm());
    //     dealii::SolverCG<VectorType> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi      precondition;
    //     precondition.initialize(system_matrix_unconstrained);
    //     //precondition.initialize(system_matrix);
    //     //solver.solve(system_matrix, trilinos_solution, system_rhs, precondition);

    //     //all_constraints.distribute(trilinos_solution);

    //     /// The use of the constrained linear operator is heavily discussed in:
    //     /// https://www.dealii.org/current/doxygen/deal.II/group__constraints.html
    //     using trilinos_vector_type = VectorType;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
    //     const auto op_amod = dealii::constrained_linear_operator(all_constraints, op_a);
    //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
    //     const auto Ct   = transpose_operator(C);
    //     // const auto Id_c = project_to_constrained_linear_operator(all_constraints, op_a);

    //     const unsigned int n_dirichlet_constraints = boundary_displacements_vector.size();
    //     dXvdXs.clear();
    //     pcout << "Solving for dXvdXs with " << n_dirichlet_constraints << "surface nodes..." << std::endl;
    //     for (unsigned int iconstraint = 0; iconstraint < n_dirichlet_constraints; iconstraint++) {

    //         VectorType unit_rhs, CtArhs, dXvdXs_i_trilinos, op_inv_CtArhs;
    //         unit_rhs.reinit(trilinos_solution);
    //         CtArhs.reinit(trilinos_solution);

    //         if (boundary_ids_vector.locally_owned_elements().is_element(iconstraint)) {
    //             const unsigned int constrained_row = boundary_ids_vector[iconstraint];
    //             unit_rhs[constrained_row] = 1.0;
    //         }
    //         CtArhs = Ct*op_a*unit_rhs;

    //         op_inv_CtArhs.reinit(CtArhs);

    //         dealii::deallog.depth_console(0);
    //         solver.solve(op_amod, op_inv_CtArhs, CtArhs, precondition);

    //         //pcout << "Surface Dirichlet constraint " << iconstraint+1 << " out of " << n_dirichlet_constraints
    //         //      << " DoF constrained: " << iconstraint
    //         //      << "    Solver converged in " << solver_control.last_step() << " iterations." << std::endl;

    //         // dXvdXs_i_trilinos.reinit(CtArhs);
    //         // dXvdXs_i_trilinos = C*op_inv_CtArhs;
    //         // dXvdXs_i_trilinos *= -1.0;
    //         // dXvdXs_i_trilinos.add(unit_rhs);

    //         // //dXvdXs.push_back(dXvdXs_i_trilinos);
    //         // dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
    //         // rw_vector.reinit(dXvdXs_i_trilinos);

    //         // dealii::LinearAlgebra::distributed::Vector<double> dXvdXs_i;
    //         // dXvdXs_i.reinit(displacement_solution);
    //         // dXvdXs_i.import(rw_vector, dealii::VectorOperation::insert);
    //         // dXvdXs.push_back(dXvdXs_i);

    //         dXvdXs_i_trilinos.reinit(CtArhs);
    //         dXvdXs_i_trilinos = C*op_inv_CtArhs;
    //         dXvdXs_i_trilinos *= -1.0;
    //         dXvdXs_i_trilinos.add(1.0,unit_rhs);
    //         dXvdXs.push_back(dXvdXs_i_trilinos);
    //     }
    //     // for (unsigned int row = 0; row < dof_handler.n_dofs(); row++) {
    //     //     bool is_locally_inhomogeneously_constrained = false;
    //     //     bool is_inhomogeneously_constrained = false;
    //     //     if (all_constraints.can_store_line(row)) {
    //     //         if (all_constraints.is_inhomogeneously_constrained(row)) {
    //     //             is_locally_inhomogeneously_constrained = true;
    //     //         }
    //     //     }
    //     //     MPI_Allreduce(&is_locally_inhomogeneously_constrained, &is_inhomogeneously_constrained, 1, MPI::BOOL, MPI_LOR, MPI_COMM_WORLD);
    //     //     if (is_inhomogeneously_constrained) {
    //     //         n_inhomogeneous_constraints++;
    //     //     }
    //     // }
    //     // dXvdXs.clear();
    //     // int i_inhomogeneous_constraints = 0;
    //     // for (unsigned int row = 0; row < dof_handler.n_dofs(); row++) {
    //     //     bool is_locally_inhomogeneously_constrained = false;
    //     //     bool is_inhomogeneously_constrained = false;
    //     //     if (all_constraints.can_store_line(row)) {
    //     //         if (all_constraints.is_inhomogeneously_constrained(row)) {
    //     //             is_locally_inhomogeneously_constrained = true;
    //     //         }
    //     //     }
    //     //     MPI_Allreduce(&is_locally_inhomogeneously_constrained, &is_inhomogeneously_constrained, 1, MPI::BOOL, MPI_LOR, MPI_COMM_WORLD);
    //     //     if (is_inhomogeneously_constrained) {
    //     //         precondition.initialize(system_matrix_unconstrained);
    //     //         VectorType unit_rhs, CtArhs, dXvdXs_i_trilinos, op_inv_CtArhs;
    //     //         unit_rhs.reinit(trilinos_solution);
    //     //         CtArhs.reinit(trilinos_solution);
    //     //         if (locally_owned_dofs.is_element(row)) {
    //     //             unit_rhs[row] = 1.0;
    //     //         }
    //     //         CtArhs = Ct*op_a*unit_rhs;

    //     //         op_inv_CtArhs.reinit(CtArhs);
    //     //         solver.solve(op_amod, op_inv_CtArhs, CtArhs, precondition);

    //     //         dXvdXs_i_trilinos.reinit(CtArhs);
    //     //         dXvdXs_i_trilinos = C*op_inv_CtArhs;
    //     //         dXvdXs_i_trilinos *= -1.0;
    //     //         dXvdXs_i_trilinos.add(unit_rhs);

    //     //         pcout << "Inhomogeneous constraint " << ++i_inhomogeneous_constraints << " out of " << n_inhomogeneous_constraints 
    //     //               << " DoF constrained: " << row
    //     //               << "    Solver converged in " << solver_control.last_step() << " iterations." << std::endl;

    //     //         dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
    //     //         rw_vector.reinit(dXvdXs_i_trilinos);

    //     //         dealii::LinearAlgebra::distributed::Vector<double> dXvdXs_i;
    //     //         dXvdXs_i.reinit(displacement_solution);
    //     //         dXvdXs_i.import(rw_vector, dealii::VectorOperation::insert);
    //     //         dXvdXs.push_back(dXvdXs_i);
    //     //     }
    //     // }
    // }
    template <int dim, typename real, typename VectorType , typename DoFHandlerType>
    unsigned int LinearElasticity<dim,real,VectorType,DoFHandlerType>::solve_linear_problem()
    {
        VectorType trilinos_solution(system_rhs);

        all_constraints.set_zero(trilinos_solution);

        //   dealii::FullMatrix<double> fullA(system_matrix.m());
        //   fullA.copy_from(system_matrix);
        //   pcout<<"Dense matrix:"<<std::endl;
        //   if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
        //   //trilinos_solution.print(std::cout, 4);
        //   system_rhs.print(std::cout, 4);

        dealii::SolverControl solver_control(5000, 1e-12 * system_rhs.l2_norm());
        dealii::SolverCG<VectorType> solver(solver_control);
        dealii::TrilinosWrappers::PreconditionJacobi      precondition;
        //precondition.initialize(system_matrix);
        //solver.solve(system_matrix, trilinos_solution, system_rhs, precondition);

        //all_constraints.distribute(trilinos_solution);

        using trilinos_vector_type = VectorType;
        using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
        const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
        //const auto op_a = dealii::linear_operator(system_matrix_unconstrained);
        const auto op_amod = dealii::constrained_linear_operator(all_constraints, op_a);
        VectorType rhs_mod
            = dealii::constrained_right_hand_side(all_constraints, op_a, system_rhs_unconstrained);
        precondition.initialize(system_matrix_unconstrained);
        solver.solve(op_amod, trilinos_solution, rhs_mod, precondition);

        
        // Should be the same as distribute
        // {
        //     VectorType trilinos_solution2(trilinos_solution);

        //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
        //     VectorType inhomogeneity_vector(trilinos_solution);
        //     for (unsigned int i=0; i<dof_handler.n_dofs(); i++) {
        //         if (inhomogeneity_vector.in_local_range(i)) {
        //             inhomogeneity_vector[i] = all_constraints.get_inhomogeneity(i);
        //         }
        //     }

        //     trilinos_solution2 = C*trilinos_solution + inhomogeneity_vector;
        //     trilinos_solution = trilinos_solution2;
        // }

        all_constraints.distribute(trilinos_solution);

        // dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
        // rw_vector.reinit(trilinos_solution);
        // displacement_solution.import(rw_vector, dealii::VectorOperation::insert);
        displacement_solution = trilinos_solution;
        return solver_control.last_step();
    }

template class LinearElasticity<PHILIP_DIM, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<PHILIP_DIM>>;
} // namespace MeshMover

} // namespace PHiLiP
