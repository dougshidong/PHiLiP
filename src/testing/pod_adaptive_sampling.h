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
#include "linear_solver/linear_solver.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "reduced_order/rom_test_location.h"
#include <deal.II/lac/householder.h>
#include <cmath>
#include <iostream>
#include <deal.II/base/function_lib.h>
#include <Eigen/Dense>
#include "reduced_order/nearest_neighbors.h"
#include "reduced_order/rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "functional/lift_drag.hpp"
#include "functional/functional.h"
#include "reduced_order/halton.h"
#include "reduced_order/min_max_scaler.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVector2d;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Class to hold information about the reduced-order solution
template <int dim, int nstate>
class AdaptiveSampling: public TestsBase
{
public:
    /// Constructor
    AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~AdaptiveSampling() {};

    mutable RowVectorXd parameter1_range;
    mutable RowVectorXd parameter2_range;
    mutable std::string parameter1_name;
    mutable std::string parameter2_name;
    mutable MatrixXd snapshot_parameters;
    mutable MatrixXd initial_rom_parameters;

    mutable std::vector<std::pair<RowVector2d, std::shared_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>>> rom_locations;

    mutable double max_error;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;

    std::shared_ptr<ProperOrthogonalDecomposition::NearestNeighbors> nearest_neighbors;

    double tolerance;

    /// Run test
    int run_test () const override;

    void placeInitialSnapshots() const;

    void placeInitialROMs() const;

    bool placeTriangulationROMs(const MatrixXd& rom_points) const;

    void updateNearestExistingROMs(const RowVector2d& parameter) const;

    RowVector2d getMaxErrorROM() const;

    dealii::LinearAlgebra::distributed::Vector<double> solveSnapshotFOM(const RowVector2d& parameter) const;

    std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVector2d& parameter) const;

    Parameters::AllParameters reinitParams(const RowVector2d& parameter) const;

    std::shared_ptr<Functional<dim,nstate,double>> functionalFactory(std::shared_ptr<DGBase<dim, double>> dg) const;

    void configureParameterSpace() const;

    void outputErrors(int iteration) const;
};

///Functional to take the integral of the solution
template <int dim, int nstate, typename real>
class BurgersRewienskiFunctional : public Functional<dim, nstate, real>
{
public:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
public:
    /// Constructor
    BurgersRewienskiFunctional(
            std::shared_ptr<PHiLiP::DGBase<dim,real>> dg_input,
            std::shared_ptr<PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType>> _physics_fad_fad,
            const bool uses_solution_values = true,
            const bool uses_solution_gradient = false)
            : PHiLiP::Functional<dim,nstate,real>(dg_input,_physics_fad_fad,uses_solution_values,uses_solution_gradient)
    {}
    //~BurgersRewienskiFunctional() = 0;

    template <typename real2>
    /// Templated volume integrand
    real2 evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
            const dealii::Point<dim,real2> &phys_coord,
            const std::array<real2,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q) const;

    /// Non-template functions to override the template classes
    real evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const dealii::Point<dim,real> &phys_coord,
            const std::array<real,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }
    /// Non-template functions to override the template classes
    FadFadType evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
            const dealii::Point<dim,FadFadType> &phys_coord,
            const std::array<FadFadType,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }
};

}
}


#endif