#ifndef __POD_ADAPTIVE_SAMPLING__
#define __POD_ADAPTIVE_SAMPLING__

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
#include "reduced_order/pod_basis_online.h"
#include "reduced_order/reduced_order_solution.h"
#include "reduced_order/full_order_solution.h"
#include "linear_solver/linear_solver.h"
#include "testing/flow_solver.h"
#include "reduced_order/rom_test_location.h"
#include <deal.II/lac/householder.h>
#include <cmath>
#include <iostream>
#include <deal.II/base/function_lib.h>
#include "testing/reduced_order_pod_adaptation.h" //For burgers rewienski functional

namespace PHiLiP {
namespace Tests {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;

/// Class to hold information about the reduced-order solution
template <int dim, int nstate>
class AdaptiveSampling: public TestsBase
{
public:
    /// Constructor
    AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~AdaptiveSampling() {};

    std::vector<double> parameter_space;

    mutable std::unordered_map<double,ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>> rom_locations;

    mutable std::vector<double> sampled_locations;

    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;

    /// Run test
    int run_test () const override;

    void initializeSampling(int n) const;

    dealii::Vector<double> polyFit(dealii::Vector<double> x, dealii::Vector<double> y, unsigned int n) const;

    dealii::Vector<double> polyVal(dealii::Vector<double> polynomial, dealii::Vector<double> x) const;

    double getMaxErrorROM() const;

    std::shared_ptr<ProperOrthogonalDecomposition::FOMSolution<dim,nstate>> solveSnapshotFOM(double parameter) const;

    std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(double parameter) const;

    Parameters::AllParameters reinitParams(double parameter) const;

};

}
}


#endif