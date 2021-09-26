#include <iostream>

#include "reduced_order.h"
#include "parameters/all_parameters.h"
#include "pod/pod.h"


namespace PHiLiP {
    namespace Tests {

        template <int dim, int nstate>
        ReducedOrder<dim, nstate>::ReducedOrder(const PHiLiP::Parameters::AllParameters *const parameters_input)
                : TestsBase::TestsBase(parameters_input)
        {}

        template <int dim, int nstate>
        int ReducedOrder<dim, nstate>::run_test() const
        {
            int num_basis = 8;

            ProperOrthogonalDecomposition::POD podtest(num_basis);
            podtest.get_full_pod_basis();

            dealii::FullMatrix<double> podbasis(podtest.svd_u->n_rows(), num_basis);
            podbasis = podtest.get_reduced_pod_basis();

            std::ofstream out_file("pod_basis_matrix.txt");
            podbasis.print_formatted(out_file, 4);
            out_file.close();

            return 0;
        }
#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace
