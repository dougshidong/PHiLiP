#ifndef __POD__
#define __POD__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>

namespace PHiLiP {
    namespace ProperOrthogonalDecomposition {

        class POD
        {
        public:

            int num_basis;
            std::unique_ptr<dealii::FullMatrix<double>> svd_u;
            std::unique_ptr<dealii::FullMatrix<double>> pod_basis;

            /// Constructor
            POD(int num_basis);

            /// Destructor
            ~POD () {};

            dealii::FullMatrix<double> get_full_pod_basis();

            dealii::FullMatrix<double> get_reduced_pod_basis();

        protected:
            const MPI_Comm mpi_communicator; ///< MPI communicator.
            dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
        };

    }
}

#endif
