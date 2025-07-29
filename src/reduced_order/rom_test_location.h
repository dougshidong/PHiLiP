#ifndef __ROM_TEST_LOCATION__
#define __ROM_TEST_LOCATION__

#include "parameters/all_parameters.h"
#include "pod_basis_base.h"
#include "reduced_order_solution.h"
#include <eigen/Eigen/Dense>
#include "test_location_base.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::RowVectorXd;

/// Class to compute and store adjoint-based error estimates
template <int dim, int nstate>
class ROMTestLocation: public TestLocationBase<dim,nstate>
{
public:
    /// Constructor
    ROMTestLocation(const RowVectorXd& parameter, std::unique_ptr<ROMSolution < dim, nstate>> rom_solution);

    /// Compute error between initial ROM and final ROM
    void compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_updated) override;

};

}
}


#endif