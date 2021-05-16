#ifndef __MESHMOVER_LINEAR_ELASTICITY_H__
#define __MESHMOVER_LINEAR_ELASTICITY_H__

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "parameters/all_parameters.h"

#include "high_order_grid.h"

namespace PHiLiP {

namespace MeshMover
{
    /** Linear elasticity mesh movement based on the deal.II example step-8 and step-42.
     */
    template <int dim = PHILIP_DIM, typename real = double>
    class LinearElasticity
    {
    /// Distributed vector of double.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>;
    /// DoFHandler
    using DoFHandlerType = dealii::DoFHandler<dim>;
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
      public:
        /// Constructor.
        LinearElasticity(
            const Triangulation &_triangulation,
            const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> mapping_fe_field,
            const DoFHandlerType &_dof_handler,
            const dealii::LinearAlgebra::distributed::Vector<int> &_boundary_ids_vector,
            const dealii::LinearAlgebra::distributed::Vector<double> &_boundary_displacements_vector);

        /// Constructor that uses information from HighOrderGrid and uses current volume_nodes from HighOrderGrid.
        LinearElasticity(
            const HighOrderGrid<dim,real> &high_order_grid,
   const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector);

        /** Evaluate and return volume displacements given boundary displacements.
         */
        VectorType get_volume_displacements();

        /** Evaluates the analytical derivatives of volume displacements with respect
         *  to surface displacements.
         */
  void evaluate_dXvdXs();

        /** Apply the analytical derivatives of volume displacements with respect
         *  to surface displacements onto a set of various right-hand sides.
         *  Note that the right-hand-side is of size n_volume_nodes.
         *  If the right-hand-side are the surface node displacements indexed in a
         *  volume node vector, the result is a displacement vector of the volume
         *  volume_nodes (which include the prescribed surface nodes).
         */
        void
        apply_dXvdXvs(std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &list_of_vectors, dealii::TrilinosWrappers::SparseMatrix &output_matrix);

        /** Apply the analytical derivatives of volume displacements with respect
         *  to surface displacements onto a set of various right-hand sides.
         *  Note that the right-hand-side is of size n_volume_nodes.
         *  If the right-hand-side are the surface node displacements indexed in a
         *  volume node vector, the result is a displacement vector of the volume
         *  volume_nodes (which include the prescribed surface nodes).
         */
        void
        apply_dXvdXvs(const dealii::LinearAlgebra::distributed::Vector<double> &input_vector, dealii::LinearAlgebra::distributed::Vector<double> &output_vector);

        /** Apply the transposed analytical derivatives of volume displacements with respect
         *  to surface displacements onto a right-hand sides.
         *  Note that the right-hand side and solution is of size n_volume_nodes.
         *  If the right-hand-side are the surface node displacements indexed in a
         *  volume node vector, the result is a displacement vector of the volume
         *  volume_nodes (which include the prescribed surface nodes).
         */
        void
        apply_dXvdXvs_transpose(
            const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
            dealii::LinearAlgebra::distributed::Vector<double> &output_vector);

        /** Current displacement solution
         */
        VectorType displacement_solution;
        /** Hanging node constraints
         */
        dealii::AffineConstraints<double> hanging_node_constraints;

        /** Vector of vectors containing the dXvdXs sensititivies.
         *  The outer vector represents the current surface index, while the inner vectors
         *  is the size of the volume mesh. Those sensitivities are distributed among the processors
         *  the same way the volume mesh volume_nodes are distributed among the processors.
         */
        //dealii::TrilinosWrappers::SparseMatrix dXvdXs;
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvdXs;

        // /** Sparse matrix containing the dXvdXs sensititivies.
        //  */
        // dealii::TrilinosWrappers::SparseMatrix<double> dXvdXs_matrix;

      private:
        /// Allocation and boundary condition setup.
        void setup_system();
        /// Assemble the system and its right-hand side.
        void assemble_system();


        /** Solve the current time step.
         *  Currently only 1 time step since it is a linear mesh mover.
         *  However, future changes will allow this to be done iteratively to
         *  obtain a nonlinear mesh mover.
         */
        void solve_timestep();

        /** Linear solver for the mesh mover.
         *  Currently uses CG with a Jacobi preconditioner.
         */
        unsigned int solve_linear_problem();

        const Triangulation &triangulation; ///< Triangulation on which this acts.
        /// MappingFEField corresponding to curved mesh.
        const std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> mapping_fe_field;
        /// Same DoFHandler as the HighOrderGrid
        const DoFHandlerType &dof_handler;
        /// Integration strength of the mesh order plus one.
        const dealii::QGauss<dim> quadrature_formula;

        /** System matrix corresponding to the unconstrained linearized elasticity problem.
         */
        dealii::TrilinosWrappers::SparseMatrix system_matrix_unconstrained;
        /** System right-hand side corresponding to the unconstrained linearized elasticity problem.
         *  Note that no body forces are present and the right-hand side is therefore zero.
         */
        dealii::LinearAlgebra::distributed::Vector<double> system_rhs_unconstrained;

        /// System matrix corresponding to linearized elasticity problem.
        dealii::TrilinosWrappers::SparseMatrix system_matrix;
        /** System right-hand side corresponding to linearized elasticity problem.
         *  Note that no body forces are present and the right-hand side is therefore zero.
         *  However, Dirichlet boundary conditions may make some RHS entries non-zero,
         *  depending on the method used to impose them.
         */
        dealii::LinearAlgebra::distributed::Vector<double> system_rhs;

        /** AffineConstraints containing boundary and hanging node constraints.
         */
        dealii::AffineConstraints<double> all_constraints;
        /** AffineConstraints containing Dirichlet boundary conditions.
         */
        dealii::AffineConstraints<double> dirichlet_boundary_constraints;

        MPI_Comm mpi_communicator; ///< MPI communicator.
        const unsigned int n_mpi_processes; ///< Number of MPI processes.
        const unsigned int this_mpi_process; ///< MPI rank.
        dealii::ConditionalOStream pcout; ///< ConditionalOStream for output.
        std::vector<dealii::types::global_dof_index> local_dofs_per_process; ///< List of number of DoFs per process.
        dealii::IndexSet locally_owned_dofs; ///< Locally owned DoFs.
        dealii::IndexSet locally_relevant_dofs; ///< Locally relevant DoFs.

        /** Global index of boundary volume_nodes that need to be constrained.
         *  Note that typically all surface boundaries "need" to be constrained.
         */
        const dealii::LinearAlgebra::distributed::Vector<int> &boundary_ids_vector;
        /** Displacement of boundary volume_nodes corresponding to boundary_ids_vector.
         */
        const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector;

        /** Transforms a std::vector<Tensor> into the corresponding distributed vector.
         */
        dealii::LinearAlgebra::distributed::Vector<double> tensor_to_vector(const std::vector<dealii::Tensor<1,dim,real>> &boundary_displacements_tensors) const;

    };
} // namespace MeshMover

} // namespace PHiLiP

#endif
