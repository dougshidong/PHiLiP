#ifndef __MESHMOVER_LINEAR_ELASTICITY_H__
#define __MESHMOVER_LINEAR_ELASTICITY_H__

#include "parameters/all_parameters.h"

#include "dg/high_order_grid.h"

namespace PHiLiP {

namespace MeshMover
{
    template <int dim = PHILIP_DIM, typename real = double, typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>, typename DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>>
    class LinearElasticity
    {
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
        /// Constructor
        LinearElasticity(
            const HighOrderGrid<dim,real,VectorType,DoFHandlerType> &high_order_grid,
			const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector);

        // ~LinearElasticity(); ///< Destructor.
        /** Evaluate and return volume displacements given boundary displacements.
         */
        VectorType get_volume_displacements();

        /** Evaluates the analytical derivatives of volume displacements with respect
         *  to surface displacements.
         */
		void evaluate_dXvdXs();

        /** Current displacement solution
         */
        VectorType displacement_solution;
        /** Hanging node constraints
         */
        dealii::AffineConstraints<double> hanging_node_constraints;

        /** Vector of vectors containing the dXvdXs sensititivies.
         *  The outer vector represents the current surface index, while the inner vectors
         *  is the size of the volume mesh. Those sensitivities are distributed among the processors
         *  the same way the volume mesh nodes are distributed among the processors.
         */
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvdXs;
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
        // void move_mesh();

        const Triangulation &triangulation; ///< Triangulation on which this acts.
        dealii::FESystem<dim> fe;
        std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> mapping_fe_field;
        const DoFHandlerType &dof_handler;
        const dealii::QGauss<dim> quadrature_formula;
        dealii::TrilinosWrappers::SparseMatrix system_matrix;
        dealii::TrilinosWrappers::MPI::Vector system_rhs;
        dealii::TrilinosWrappers::SparseMatrix system_matrix_unconstrained;
        dealii::TrilinosWrappers::MPI::Vector system_rhs_unconstrained;

        dealii::AffineConstraints<double> all_constraints;
        dealii::AffineConstraints<double> dirichlet_boundary_constraints;

        MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
        dealii::ConditionalOStream pcout;
        std::vector<dealii::types::global_dof_index> local_dofs_per_process;
        dealii::IndexSet locally_owned_dofs;
        dealii::IndexSet locally_relevant_dofs;

        const dealii::LinearAlgebra::distributed::Vector<int> &boundary_ids_vector;
        const dealii::LinearAlgebra::distributed::Vector<double> &boundary_displacements_vector;

    };
} // namespace MeshMover

} // namespace PHiLiP

#endif
