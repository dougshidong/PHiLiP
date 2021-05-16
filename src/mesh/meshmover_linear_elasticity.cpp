#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/dofs/dof_tools.h>

//#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
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

    // template <int dim, typename real>
    // LinearElasticity<dim,real>::LinearElasticity(
    //     const HighOrderGrid<dim,real> &high_order_grid,
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

    template <int dim, typename real>
    LinearElasticity<dim,real>::LinearElasticity(
        const HighOrderGrid<dim,real> &high_order_grid,
        const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector)
      : LinearElasticity<dim,real> (
          *(high_order_grid.triangulation),
          high_order_grid.mapping_fe_field,
          high_order_grid.dof_handler_grid,
          high_order_grid.surface_to_volume_indices,
          boundary_displacements_vector)
    { }

    template <int dim, typename real>
    LinearElasticity<dim,real>::LinearElasticity(
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

    // template <int dim, typename real>
    // LinearElasticity<dim,real>::LinearElasticity(
    //     const HighOrderGrid<dim,real> &high_order_grid,
    //     const std::vector<dealii::Tensor<1,dim,real>> &boundary_displacements_tensors)
    //   : LinearElasticity(high_order_grid, boundary_displacements_vector(tensor_to_vector(boundary_displacements_tensors))
    // { }

    template <int dim, typename real>
    dealii::LinearAlgebra::distributed::Vector<double> LinearElasticity<dim,real>::
    tensor_to_vector(const std::vector<dealii::Tensor<1,dim,real>> &boundary_displacements_tensors) const
    {
        (void) boundary_displacements_tensors;
        dealii::LinearAlgebra::distributed::Vector<double> boundary_displacements_vector;
        return boundary_displacements_vector;
    }

    //template <int dim, typename real>
    //LinearElasticity<dim,real>::~LinearElasticity() { dof_handler.clear(); }

    template <int dim, typename real>
    dealii::LinearAlgebra::distributed::Vector<real>
    LinearElasticity<dim,real>::get_volume_displacements()
    {
        pcout << "Solving linear elasticity problem for volume displacements..." << std::endl;
        solve_timestep();
        // displacement_solution = 0;
        // all_constraints.distribute(displacement_solution);
        displacement_solution.update_ghost_values();
        return displacement_solution;
    }
    template <int dim, typename real>
    void LinearElasticity<dim,real>::setup_system()
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
        hanging_node_constraints.close();

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
                                        hanging_node_constraints,
                                        //all_constraints,
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

        // pcout << "    Number of active cells: " << triangulation.n_active_cells() << std::endl;
        // pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
    }
    template <int dim, typename real>
    void LinearElasticity<dim,real>::assemble_system()
    {
        pcout << "    Assembling MeshMover::LinearElasticity system..." << std::endl;

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

        //std::vector<double> youngs_modulus(n_q_points, 1.0);
        //std::vector<double> poissons_ratio(n_q_points, 0.4);

        std::vector<double> youngs_modulus(n_q_points, 1.0);
        std::vector<double> poissons_ratio(n_q_points, 0.1);

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

            double volume = 0.0;
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                volume += fe_values.JxW(q_point);
            }

            //lame_lambda.value_list(fe_values.get_quadrature_points(), lame_lambda_values);
            //lame_mu.value_list(fe_values.get_quadrature_points(), lame_mu_values);

            for (unsigned int itest = 0; itest < dofs_per_cell; ++itest) {

                const unsigned int component_test = fe_system.system_to_component_index(itest).first;

                for (unsigned int itrial = 0; itrial < dofs_per_cell; ++itrial) {

                    const unsigned int component_trial = fe_system.system_to_component_index(itrial).first;

                    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                        const double E = youngs_modulus[q_point] / volume;
                        const double nu = poissons_ratio[q_point];
                        lame_lambda_values[q_point] = E*nu/((1.0+nu)*(1-2.0*nu));
                        lame_mu_values[q_point] = 0.5*E/(1.0+nu);
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
            //all_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs, use_inhomogeneities_for_rhs);
            hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs, use_inhomogeneities_for_rhs);
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
        const auto &partitionner = boundary_ids_vector.get_partitioner();
        MPI_Barrier(MPI_COMM_WORLD);
        for (unsigned int isurf = 0; isurf < boundary_ids_vector.size(); ++isurf) {
            const bool is_accessible = partitionner->in_local_range(isurf) || partitionner->is_ghost_entry(isurf);
            if (is_accessible) {
                const unsigned int iglobal_row = boundary_ids_vector[isurf];
                const double dirichlet_value = boundary_displacements_vector[isurf];

                system_matrix.clear_row(iglobal_row,1.0);
                system_rhs[iglobal_row] = dirichlet_value;
            }
        }
        // Until deal.II accepts the pull request to fix TrilinosWrappers::SparseMatrix::clear_row(row,new_diag_value)
        // Manually set the value of the diagonal to 1.0 here.
        for (unsigned int isurf = 0; isurf < boundary_ids_vector.size(); ++isurf) {
            const bool is_accessible = partitionner->in_local_range(isurf) || partitionner->is_ghost_entry(isurf);
            if (is_accessible) {
                const unsigned int iglobal_row = boundary_ids_vector[isurf];
                system_matrix.set(iglobal_row,iglobal_row,1.0);
            }
        }
        system_matrix.compress(dealii::VectorOperation::insert);
        system_rhs.compress(dealii::VectorOperation::insert);
        system_matrix_unconstrained.compress(dealii::VectorOperation::insert);
        system_rhs_unconstrained.compress(dealii::VectorOperation::insert);
    }
    template <int dim, typename real>
    void LinearElasticity<dim,real>::solve_timestep()
    {
        assemble_system();
        apply_dXvdXvs(system_rhs, displacement_solution);
        //const unsigned int n_iterations = solve_linear_problem();
        //pcout << "    Solver converged in " << n_iterations << " iterations." << std::endl;
    }

    template <int dim, typename real>
    void
    LinearElasticity<dim,real>
    ::apply_dXvdXvs(
        const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
        dealii::LinearAlgebra::distributed::Vector<double> &output_vector)
    {
        pcout << "Applying [dXvdXs] onto a vector..." << std::endl;
        assert(input_vector.size() == output_vector.size());

        double input_vector_norm = input_vector.l2_norm();
        if (input_vector_norm == 0.0) {
            pcout << "Zero input vector. Zero output vector." << std::endl;
            output_vector = 0.0;
            return;
        }

        assemble_system();

        const bool log_history = (this_mpi_process == 0);
        dealii::SolverControl solver_control(20000, 1e-14 * input_vector_norm, log_history);
        //dealii::SolverControl solver_control(20000, 1e-14, log_history);
        solver_control.log_frequency(100);
        const int max_n_tmp_vectors=200;
        const bool right_preconditioning=true;
        const bool use_default_residual=true;
        const bool force_re_orthogonalization=false;
        dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_settings(max_n_tmp_vectors, right_preconditioning, use_default_residual, force_re_orthogonalization);
        dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control, gmres_settings);

        //dealii::TrilinosWrappers::PreconditionJacobi      precondition;
        //precondition.initialize(system_matrix);

        //dealii::TrilinosWrappers::PreconditionILU  precondition;
        //const int ilu_fill = 2;
        //dealii::TrilinosWrappers::PreconditionILU::AdditionalData precond_settings(ilu_fill, 0., 1.0, 1);
        //precondition.initialize(system_matrix, precond_settings);

        dealii::TrilinosWrappers::PreconditionILUT  precondition;
        const unsigned int ilut_fill=50;
        const double ilut_drop=1e-15;
        const double ilut_atol=1e-6;
        const double ilut_rtol=1.00001;
        const unsigned int overlap=1;
        dealii::TrilinosWrappers::PreconditionILUT::AdditionalData precond_settings(ilut_drop, ilut_fill, ilut_atol, ilut_rtol, overlap);
        precondition.initialize(system_matrix, precond_settings);


        //const double 	omega = 1;
        //const double 	min_diagonal = 1e-8;
        //const unsigned int 	overlap = 1;
        //const unsigned int 	n_sweeps = 1;
        //dealii::TrilinosWrappers::PreconditionSSOR::AdditionalData precond_settings(omega, min_diagonal, overlap, n_sweeps);
        //dealii::TrilinosWrappers::PreconditionSSOR  precondition;
        //precondition.initialize(system_matrix, precond_settings);

        using trilinos_vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
        using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
        const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix);

        const auto &rhs_vector = input_vector;
        output_vector = input_vector;
        //output_vector = 0.0;//input_vector;

        // Solve modified system.
        dealii::deallog.depth_console(2);
        solver.solve(op_a, output_vector, rhs_vector, precondition);

        pcout << "dXvdXvs Solver took " << solver_control.last_step() << " steps. "
              << "Residual: " << solver_control.last_value() << ". "
              << std::endl;

        solver.solve(op_a, output_vector, rhs_vector, precondition);

        pcout << "dXvdXvs Solver took " << solver_control.last_step() << " steps. "
              << "Residual: " << solver_control.last_value() << ". "
              << std::endl;

        if (solver_control.last_check() != dealii::SolverControl::State::success) {
            pcout << "Failed to converge." << std::endl;
            std::abort();
        }
    }

    template <int dim, typename real>
    void
    LinearElasticity<dim,real>
    ::apply_dXvdXvs(
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

        const dealii::IndexSet col_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_cols);

        output_matrix.reinit(row_part, col_part, full_sp, mpi_communicator);

        //dealii::TrilinosWrappers::PreconditionJacobi      precondition;
        //precondition.initialize(system_matrix);

        //dealii::TrilinosWrappers::PreconditionILU  precondition;
        //const int ilu_fill = 2;
        //dealii::TrilinosWrappers::PreconditionILU::AdditionalData precond_settings(ilu_fill, 0., 1.0, 1);
        //precondition.initialize(system_matrix, precond_settings);

        dealii::TrilinosWrappers::PreconditionILUT  precondition;
        const unsigned int ilut_fill=50;
        const double ilut_drop=0.0;//1e-15;
        const double ilut_atol=0.0;//1e-6;
        const double ilut_rtol=1.0;//1.00001;
        const unsigned int overlap=1;
        dealii::TrilinosWrappers::PreconditionILUT::AdditionalData precond_settings(ilut_drop, ilut_fill, ilut_atol, ilut_rtol, overlap);
        precondition.initialize(system_matrix, precond_settings);


        //const double 	omega = 1;
        //const double 	min_diagonal = 1e-8;
        //const unsigned int 	overlap = 1;
        //const unsigned int 	n_sweeps = 1;
        //dealii::TrilinosWrappers::PreconditionSSOR::AdditionalData precond_settings(omega, min_diagonal, overlap, n_sweeps);
        //dealii::TrilinosWrappers::PreconditionSSOR  precondition;
        //precondition.initialize(system_matrix, precond_settings);

        using trilinos_vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
        using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
        const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix);

        dXvdXs.clear();
        pcout << "Applying for [dXvdXs] onto " << list_of_vectors.size() << " vectors..." << std::endl;

        unsigned int col = 0;
        dealii::LinearAlgebra::distributed::Vector<double> output_vector;
        output_vector.reinit(list_of_vectors[0]);
        for (auto &input_vector: list_of_vectors) {

            pcout << " Vector " << col << " out of " << list_of_vectors.size() << std::endl;

            dealii::deallog.depth_console(0);

            double input_vector_norm = input_vector.l2_norm();
            if (input_vector_norm == 0.0) {
                pcout << "Zero input vector. Zero output vector." << std::endl;
                output_vector = 0.0;
            } else {
                const bool log_history = (this_mpi_process == 0);
                dealii::SolverControl solver_control(20000, 1e-14 * input_vector_norm, log_history);
                //dealii::SolverControl solver_control(20000, 1e-14, log_history);
                solver_control.log_frequency(100);
                const int max_n_tmp_vectors=200;
                const bool right_preconditioning=true;
                const bool use_default_residual=true;
                const bool force_re_orthogonalization=false;
                dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_settings(max_n_tmp_vectors, right_preconditioning, use_default_residual, force_re_orthogonalization);
                dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control, gmres_settings);

                //dealii::SolverBicgstab<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);

                dealii::deallog.depth_console(2);
                solver.solve(op_a, output_vector, input_vector, precondition);
                pcout << "dXvdXvs Solver took " << solver_control.last_step() << " steps. "
                      << "Residual: " << solver_control.last_value() << ". "
                      << std::endl;

                solver.solve(op_a, output_vector, input_vector, precondition);
                pcout << "dXvdXvs Solver took " << solver_control.last_step() << " steps. "
                      << "Residual: " << solver_control.last_value() << ". "
                      << std::endl;
            }

            dXvdXs.push_back(output_vector);

            for (const auto &row: dof_handler.locally_owned_dofs()) {
                output_matrix.set(row, col, dXvdXs[col][row]);
            }
            col++;
        }
        output_matrix.compress(dealii::VectorOperation::insert);

    }

    template <int dim, typename real>
    void
    LinearElasticity<dim,real>
    ::apply_dXvdXvs_transpose(
        const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
        dealii::LinearAlgebra::distributed::Vector<double> &output_vector)
    {
        pcout << "Applying [transpose(dXvdXvs)] onto a vector..." << std::endl;

        double input_vector_norm = input_vector.l2_norm();
        if (input_vector_norm == 0.0) {
            pcout << "Zero input vector. Zero output vector." << std::endl;
            output_vector = 0.0;
            return;
        }

        assemble_system();

        const bool log_history = (this_mpi_process == 0);
        dealii::SolverControl solver_control(20000, 1e-14 * input_vector_norm, log_history);
        //dealii::SolverControl solver_control(20000, 1e-14, log_history);
        solver_control.log_frequency(100);
        const int max_n_tmp_vectors=200;
        const bool right_preconditioning=true;
        const bool use_default_residual=true;
        const bool force_re_orthogonalization=false;
        dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_settings(max_n_tmp_vectors, right_preconditioning, use_default_residual, force_re_orthogonalization);
        dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control, gmres_settings);

        //dealii::TrilinosWrappers::PreconditionJacobi      precondition;
        //precondition.initialize(system_matrix);

        // dealii::TrilinosWrappers::PreconditionILU  precondition;
        // const int ilu_fill = 2;
        // dealii::TrilinosWrappers::PreconditionILU::AdditionalData precond_settings(ilu_fill, 0., 1.0, 1);
        // precondition.initialize(system_matrix, precond_settings);

        dealii::TrilinosWrappers::PreconditionILUT  precondition;
        const unsigned int ilut_fill=50;
        const double ilut_drop=1e-15;
        const double ilut_atol=1e-6;
        const double ilut_rtol=1.00001;
        const unsigned int overlap=1;
        dealii::TrilinosWrappers::PreconditionILUT::AdditionalData precond_settings(ilut_drop, ilut_fill, ilut_atol, ilut_rtol, overlap);
        precondition.initialize(system_matrix, precond_settings);

        //const double 	omega = 1;
        //const double 	min_diagonal = 1e-8;
        //const unsigned int 	overlap = 1;
        //const unsigned int 	n_sweeps = 1;
        //dealii::TrilinosWrappers::PreconditionSSOR::AdditionalData precond_settings(omega, min_diagonal, overlap, n_sweeps);
        //dealii::TrilinosWrappers::PreconditionSSOR  precondition;
        //precondition.initialize(system_matrix, precond_settings);

        using trilinos_vector_type = VectorType;
        using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
        const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix);
        const auto op_at = dealii::transpose_operator(op_a);

        // Solve system.
        dealii::deallog.depth_console(2);
        output_vector = input_vector;
        solver.solve(op_at, output_vector, input_vector, precondition);
        pcout << "dXvdXvs_Transpose Solver took " << solver_control.last_step() << " steps. "
              << "Residual: " << solver_control.last_value() << ". "
              << std::endl;
        solver.solve(op_at, output_vector, input_vector, precondition);
        pcout << "dXvdXvs_Transpose Solver took " << solver_control.last_step() << " steps. "
              << "Residual: " << solver_control.last_value() << ". "
              << std::endl;
        if (solver_control.last_check() != dealii::SolverControl::State::success) {
            pcout << "Failed to converge." << std::endl;
            std::abort();
        }


    }

    // template <int dim, typename real>
    // unsigned int LinearElasticity<dim,real>::solve_linear_problem()
    // {
    //     displacement_solution.reinit(system_rhs);

    //     dealii::SolverControl solver_control(20000, 1e-14 * system_rhs.l2_norm());
    //     dealii::SolverGMRES<VectorType> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi precondition;
    //     precondition.initialize(system_matrix);

    //     using trilinos_vector_type = VectorType;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix);

    //     solver.solve(op_a, displacement_solution, system_rhs, precondition);

    //     return solver_control.last_step();
    // }

    // template <int dim, typename real>
    // void
    // LinearElasticity<dim,real>
    // ::apply_dXvdXvs(
    //     const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
    //     dealii::LinearAlgebra::distributed::Vector<double> &output_vector)
    // {
    //     pcout << "Applying [dXvdXs] onto a vector..." << std::endl;
    //     assert(input_vector.size() == output_vector.size());

    //     assemble_system();

    //     dealii::SolverControl solver_control(20000, 1e-14 * system_rhs.l2_norm());
    //     dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi      precondition;
    //     precondition.initialize(system_matrix_unconstrained);

    //     /// The use of the constrained linear operator is heavily discussed in:
    //     /// https://www.dealii.org/current/doxygen/deal.II/group__constraints.html
    //     /// Given affine constraints such that x = C y + k
    //     /// where C describes the homogeneous part of the linear constraints stored in an AffineConstraints object
    //     /// and the vector k is the vector of corresponding inhomogeneities
    //     ///
    //     /// Eg. Dirichlet BC's would have zero-rows in C and non-zero rows in k
    //     /// and hanging-nodes would be linearly constrained through non-zero rows within C.
    //     ///
    //     /// 1.  (Ct A_unconstrained C + Id_c) y = Ct (b - Ak)
    //     /// 2.  x = C y + k
    //     ///
    //     /// b are the forces, which == 0
    //     /// k are the inhomogeneous
    //     /// Id_c Identity on the subspace of constrained degrees of freedom.
    //     /// 
    //     /// The above steps 1. and 2. solve the real constrained system A_constrained x = b_constrained
    //     /// Although possible to assemble and solve, we will be interested in the derivative with respect
    //     /// to the inhomogeneity vector k, which is more easily recoverable through formulation 1. and 2.,
    //     /// than the assembly of the constrained system.
    //     ///
    //     /// y = - inverse(Ct A_unconstrained C + Id_c) Ct A k
    //     /// x = - C inverse(Ct A_unconstrained C + Id_c) Ct A k + k
    //     /// dx/dk = - C inverse(Ct A_unconstrained C + Id_c) Ct A + I
    //     using trilinos_vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
    //     const auto op_amod = dealii::constrained_linear_operator(all_constraints, op_a);
    //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
    //     const auto Ct   = transpose_operator(C);
    //     const auto Id_c = project_to_constrained_linear_operator(all_constraints, op_a);

    //     const auto &rhs_vector = input_vector;

    //     // Build RHS.
    //     dealii::LinearAlgebra::distributed::Vector<double> CtArhs;
    //     CtArhs.reinit(rhs_vector);
    //     CtArhs = Ct*op_a*rhs_vector;

    //     // Solution.
    //     dealii::LinearAlgebra::distributed::Vector<double> op_inv_CtArhs(rhs_vector);
    //     all_constraints.set_zero(op_inv_CtArhs);

    //     // Solve modified system.
    //     dealii::deallog.depth_console(0);
    //     solver.solve(op_amod, op_inv_CtArhs, CtArhs, precondition);

    //     // Apply boundary condition
    //     output_vector = C*op_inv_CtArhs;
    //     output_vector *= -1.0;
    //     output_vector += rhs_vector;
    // }

    // template <int dim, typename real>
    // void
    // LinearElasticity<dim,real>
    // ::apply_dXvdXvs(
    //     std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &list_of_vectors,
    //     dealii::TrilinosWrappers::SparseMatrix &output_matrix)
    // {
    //     assemble_system();

    //     const unsigned int n_rows = dof_handler.n_dofs();
    //     const unsigned int n_cols = list_of_vectors.size();
    //     //const unsigned int max_per_row = n_cols;

    //     const dealii::IndexSet &row_part = dof_handler.locally_owned_dofs();
    //     dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    //     dealii::DynamicSparsityPattern full_dsp(n_rows, n_cols, row_part);
    //     for (const auto &i_row: row_part) {
    //         for (unsigned int i_col = 0; i_col < n_cols; ++i_col) {
    //             full_dsp.add(i_row, i_col);
    //         }
    //     }
    //     dealii::SparsityTools::distribute_sparsity_pattern(full_dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

    //     dealii::SparsityPattern full_sp;
    //     full_sp.copy_from(full_dsp);

    //     const dealii::IndexSet col_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_cols);

    //     output_matrix.reinit(row_part, col_part, full_sp, mpi_communicator);

    //     dealii::SolverControl solver_control(20000, 1e-14 * system_rhs.l2_norm());
    //     dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi      precondition;
    //     precondition.initialize(system_matrix_unconstrained);

    //     /// The use of the constrained linear operator is heavily discussed in:
    //     /// https://www.dealii.org/current/doxygen/deal.II/group__constraints.html
    //     /// Given affine constraints such that x = C y + k
    //     /// where C describes the homogeneous part of the linear constraints stored in an AffineConstraints object
    //     /// and the vector k is the vector of corresponding inhomogeneities
    //     ///
    //     /// Eg. Dirichlet BC's would have zero-rows in C and non-zero rows in k
    //     /// and hanging-nodes would be linearly constrained through non-zero rows within C.
    //     ///
    //     /// 1.  (Ct A_unconstrained C + Id_c) y = Ct (b - Ak)
    //     /// 2.  x = C y + k
    //     ///
    //     /// b are the forces, which == 0
    //     /// k are the inhomogeneous
    //     /// Id_c Identity on the subspace of constrained degrees of freedom.
    //     /// 
    //     /// The above steps 1. and 2. solve the real constrained system A_constrained x = b_constrained
    //     /// Although possible to assemble and solve, we will be interested in the derivative with respect
    //     /// to the inhomogeneity vector k, which is more easily recoverable through formulation 1. and 2.,
    //     /// than the assembly of the constrained system.
    //     ///
    //     /// y = - inverse(Ct A_unconstrained C + Id_c) Ct A k
    //     /// x = - C inverse(Ct A_unconstrained C + Id_c) Ct A k + k
    //     /// dx/dk = - C inverse(Ct A_unconstrained C + Id_c) Ct A + I
    //     using trilinos_vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
    //     const auto op_amod = dealii::constrained_linear_operator(all_constraints, op_a);
    //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
    //     const auto Ct   = transpose_operator(C);
    //     const auto Id_c = project_to_constrained_linear_operator(all_constraints, op_a);

    //     dXvdXs.clear();
    //     pcout << "Applying for [dXvdXs] onto " << list_of_vectors.size() << " vectors..." << std::endl;

    //     unsigned int col = 0;
    //     for (auto &rhs_vector: list_of_vectors) {

    //         // Build RHS.
    //         dealii::LinearAlgebra::distributed::Vector<double> CtArhs;
    //         CtArhs.reinit(rhs_vector);
    //         CtArhs = Ct*op_a*rhs_vector;

    //         // Solution.
    //         dealii::LinearAlgebra::distributed::Vector<double> op_inv_CtArhs(rhs_vector);
    //         all_constraints.set_zero(op_inv_CtArhs);

    //         // Solve modified system.
    //         dealii::deallog.depth_console(0);
    //         solver.solve(op_amod, op_inv_CtArhs, CtArhs, precondition);

    //         // Apply boundary condition
    //         dealii::LinearAlgebra::distributed::Vector<double> dXvdXs_i_trilinos;
    //         dXvdXs_i_trilinos.reinit(CtArhs);
    //         dXvdXs_i_trilinos = C*op_inv_CtArhs;
    //         dXvdXs_i_trilinos *= -1.0;
    //         dXvdXs_i_trilinos += rhs_vector;

    //         dXvdXs.push_back(dXvdXs_i_trilinos);

    //         for (const auto &row: dof_handler.locally_owned_dofs()) {
    //             output_matrix.set(row, col, dXvdXs[col][row]);
    //         }
    //         col++;
    //     }
    //     output_matrix.compress(dealii::VectorOperation::insert);

    // }

    // template <int dim, typename real>
    // void
    // LinearElasticity<dim,real>
    // ::apply_dXvdXvs_transpose(
    //     const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
    //     dealii::LinearAlgebra::distributed::Vector<double> &output_vector)
    // {
    //     pcout << "Applying [transpose(dXvdXvs)] onto a vector..." << std::endl;
    //     assemble_system();

    //     dealii::SolverControl solver_control(20000, 1e-14 * system_rhs.l2_norm());
    //     dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi      precondition;
    //     precondition.initialize(system_matrix_unconstrained);

    //     /// The use of the constrained linear operator is heavily discussed in:
    //     /// https://www.dealii.org/current/doxygen/deal.II/group__constraints.html
    //     /// x = - C inverse(Ct A_unconstrained C + Id_c) Ct A k + k
    //     /// dx/dk = - C inverse(Ct A_unconstrained C + Id_c) Ct A + I
    //     /// transpose(dx/dk) = - transpose(A) C transpose(inverse(Ct A_unconstrained C + Id_c)) Ct + I
    //     using trilinos_vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
    //     const auto op_at = transpose_operator(op_a);
    //     const auto op_amod_trans = transpose_operator(dealii::constrained_linear_operator(all_constraints, op_a));
    //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
    //     const auto Ct   = transpose_operator(C);
    //     const auto Id_c = project_to_constrained_linear_operator(all_constraints, op_a);

    //     // Build RHS.
    //     dealii::LinearAlgebra::distributed::Vector<double> Ctrhs;
    //     Ctrhs.reinit(input_vector);
    //     Ctrhs = Ct*input_vector;

    //     // Solution.
    //     dealii::LinearAlgebra::distributed::Vector<double> op_inv_Ctrhs(input_vector);
    //     all_constraints.set_zero(op_inv_Ctrhs);

    //     // Solve modified system.
    //     dealii::deallog.depth_console(0);
    //     solver.solve(op_amod_trans, op_inv_Ctrhs, Ctrhs, precondition);

    //     // Apply boundary condition
    //     output_vector.reinit(Ctrhs);
    //     output_vector = op_at*C*op_inv_Ctrhs;
    //     output_vector *= -1.0;
    //     output_vector += input_vector;

    //     //output_vector.compress(dealii::VectorOperation::insert);

    // }

    template <int dim, typename real>
    void LinearElasticity<dim,real>::evaluate_dXvdXs()
    {
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> unit_rhs_vector;
        const unsigned int n_dirichlet_constraints = boundary_displacements_vector.size();
        dXvdXs.clear();
        pcout << "Solving for dXvdXs with " << n_dirichlet_constraints << " surface nodes..." << std::endl;
        pcout << "*******************************************************" << std::endl;
        pcout << "Are you sure that you need to for the dXvdXs matrix...?" << std::endl;
        pcout << "Ideally, you would apply dXvdXs onto a vector such that" << std::endl;
        pcout << "only 1 linear system needs to be solved.               " << std::endl;
        pcout << "*******************************************************" << std::endl;
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
        apply_dXvdXvs(unit_rhs_vector, dXvdXs_matrix);
    }

    // template <int dim, typename real>
    // void LinearElasticity<dim,real>::evaluate_dXvdXs()
    // {
    //     VectorType trilinos_solution(system_rhs);

    //     all_constraints.set_zero(trilinos_solution);

    //     dealii::SolverControl solver_control(20000, 1e-14 * system_rhs.l2_norm());
    //     dealii::SolverGMRES<VectorType> solver(solver_control);
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
    // template <int dim, typename real>
    // unsigned int LinearElasticity<dim,real>::solve_linear_problem()
    // {
    //     VectorType trilinos_solution(system_rhs);

    //     all_constraints.set_zero(trilinos_solution);

    //     //   dealii::FullMatrix<double> fullA(system_matrix.m());
    //     //   fullA.copy_from(system_matrix);
    //     //   pcout<<"Dense matrix:"<<std::endl;
    //     //   if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
    //     //   //trilinos_solution.print(std::cout, 4);
    //     //   system_rhs.print(std::cout, 4);

    //     dealii::SolverControl solver_control(20000, 1e-14 * system_rhs.l2_norm());
    //     dealii::SolverGMRES<VectorType> solver(solver_control);
    //     dealii::TrilinosWrappers::PreconditionJacobi      precondition;
    //     //precondition.initialize(system_matrix);
    //     //solver.solve(system_matrix, trilinos_solution, system_rhs, precondition);

    //     //all_constraints.distribute(trilinos_solution);

    //     using trilinos_vector_type = VectorType;
    //     using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    //     const auto op_a = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(system_matrix_unconstrained);
    //     //const auto op_a = dealii::linear_operator(system_matrix_unconstrained);
    //     const auto op_amod = dealii::constrained_linear_operator(all_constraints, op_a);
    //     VectorType rhs_mod
    //         = dealii::constrained_right_hand_side(all_constraints, op_a, system_rhs_unconstrained);
    //     precondition.initialize(system_matrix_unconstrained);
    //     solver.solve(op_amod, trilinos_solution, rhs_mod, precondition);

    //     
    //     // Should be the same as distribute
    //     // {
    //     //     VectorType trilinos_solution2(trilinos_solution);

    //     //     const auto C    = distribute_constraints_linear_operator(all_constraints, op_a);
    //     //     VectorType inhomogeneity_vector(trilinos_solution);
    //     //     for (unsigned int i=0; i<dof_handler.n_dofs(); i++) {
    //     //         if (inhomogeneity_vector.in_local_range(i)) {
    //     //             inhomogeneity_vector[i] = all_constraints.get_inhomogeneity(i);
    //     //         }
    //     //     }

    //     //     trilinos_solution2 = C*trilinos_solution + inhomogeneity_vector;
    //     //     trilinos_solution = trilinos_solution2;
    //     // }

    //     all_constraints.distribute(trilinos_solution);

    //     // dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
    //     // rw_vector.reinit(trilinos_solution);
    //     // displacement_solution.import(rw_vector, dealii::VectorOperation::insert);
    //     displacement_solution = trilinos_solution;
    //     return solver_control.last_step();
    // }

template class LinearElasticity<PHILIP_DIM, double>;
} // namespace MeshMover

} // namespace PHiLiP
