#ifndef __ROM_IMPORT_HELPER_FUNCTIONS_H__
#define __ROM_IMPORT_HELPER_FUNCTIONS_H__

#include "rom_import_helper_functions.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/// Read snapshot locations from the text file
bool getSnapshotParamsFromFile(Eigen::MatrixXd& snapshot_parameters, std::string path);

/// Place 400 distributed points across the parameter domain for error sampling
void getROMPoints(Eigen::MatrixXd& rom_points, const Parameters::AllParameters *const all_parameters);

} // End of Tests namespace
} // End of PHiLiP namespace

#endif