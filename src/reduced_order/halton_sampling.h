#ifndef __HALTON_SAMPLING__
#define __HALTON_SAMPLING__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"
#include "rom_test_location.h"
#include <eigen/Eigen/Dense>
#include "nearest_neighbors.h"
#include "adaptive_sampling_base.h"

namespace PHiLiP {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Halton sampling
/*
Based on the work in Donovan Blais' thesis:
Goal-Oriented Adaptive Sampling for Projection-Based Reduced-Order Models, 2022

This is a sampling procedure similar to the adaptive sampling, which instead uses a halton sequence to select the snapshot parameter locations.
From Wikipedia (https://en.wikipedia.org/wiki/Halton_sequence):
In statistics, Halton sequences are sequences used to generate points in space for numerical methods such as Monte Carlo simulations.
The Halton sequence is constructed according to a deterministic method that uses coprime numbers as its bases.
*/
template <int dim, int nstate>
class HaltonSampling: public AdaptiveSamplingBase<dim,nstate>
{
public:
    /// Constructor
    HaltonSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~HaltonSampling() {};

    /// Run Sampling Procedure
    int run_sampling () const override;
};

}


#endif