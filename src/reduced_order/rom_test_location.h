#ifndef __ROM_TEST_LOCATION__
#define __ROM_TEST_LOCATION__

#include <fstream>
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod_interfaces.h"
#include "reduced_order_solution.h"
#include "full_order_solution.h"
#include "linear_solver/linear_solver.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_LinearProblem.h>
#include "Amesos.h"
#include "Amesos_BaseSolver.h"



namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::RowVector2d;


/// Class to hold information about the reduced-order solution
template <int dim, int nstate>
class ROMTestLocation
{
public:
    /// Constructor
    ROMTestLocation(RowVector2d parameter, std::shared_ptr<ROMSolution < dim, nstate>> rom_solution);

    /// Copy Constructor
    //ROMTestLocation(const ROMTestLocation& rom_test_location) = default;

    ///Assignment operator
    //ROMTestLocation& operator= (const ROMTestLocation& rom_test_location) = default;

    /// Destructor
    ~ROMTestLocation() {};

    void compute_FOM_to_initial_ROM_error();

    void compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod_updated);

    void compute_total_error();

    RowVector2d parameter;

    std::shared_ptr<ROMSolution<dim, nstate>> rom_solution;

    double fom_to_initial_rom_error;

    double initial_rom_to_final_rom_error;

    double total_error;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

};

}
}


#endif