#ifndef __HROM_ERROR_POST_SAMPLING_H__
#define __HROM_ERROR_POST_SAMPLING_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include "dg/dg_base.hpp"
#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Find the "true" error between the FOM and a HROM (with ECSW hyperreduction) built from the snapshots after
/// an adaptive sampling procedure has been run.
/// NOTE: The folder the test reads from should only contain the outputted files from the last iteration of
/// the adaptive sampling procedure. It should include one text file beginning with "snapshot_table", one
/// beginning with "solution_snapshots", and one beginning with "weights" which contains the last ECSW weights
/// found in the adaptive sampling procedure.
template <int dim, int nstate>
class HROMErrorPostSampling: public TestsBase
{
public:
    /// Constructor.
    HROMErrorPostSampling(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(std::string path) const;

    /// Read ECSW weights from the text file 
    bool getWeightsFromFile(std::shared_ptr<DGBase<dim,double>> &dg) const;

    /// Evaluate and output the "true" error at ROM Points
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    /// Matrix of error sampling points
    mutable MatrixXd rom_points;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
