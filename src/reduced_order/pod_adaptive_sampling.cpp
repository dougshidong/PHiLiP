#include "pod_adaptive_sampling.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <cmath>
#include "rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "halton.h"
#include "min_max_scaler.h"

namespace PHiLiP {

template<int dim, int nstate>
AdaptiveSampling<dim, nstate>::AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                const dealii::ParameterHandler &parameter_handler_input)
        : AdaptiveSamplingBase<dim, nstate>(parameters_input, parameter_handler_input)
{}

template <int dim, int nstate>
int AdaptiveSampling<dim, nstate>::run_sampling() const
{
    this->pcout << "Starting adaptive sampling process" << std::endl;

    this->placeInitialSnapshots();
    this->current_pod->computeBasis();

    MatrixXd rom_points = this->nearest_neighbors->kPairwiseNearestNeighborsMidpoint();
    this->pcout << rom_points << std::endl;

    placeROMLocations(rom_points);

    RowVectorXd max_error_params = this->getMaxErrorROM();
    int iteration = 0;

    while(this->max_error > this->all_parameters->reduced_order_param.adaptation_tolerance){

        this->outputIterationData(std::to_string(iteration));

        this->pcout << "Sampling snapshot at " << max_error_params << std::endl;
        dealii::LinearAlgebra::distributed::Vector<double> fom_solution = this->solveSnapshotFOM(max_error_params);
        this->snapshot_parameters.conservativeResize(this->snapshot_parameters.rows()+1, this->snapshot_parameters.cols());
        this->snapshot_parameters.row(this->snapshot_parameters.rows()-1) = max_error_params;
        this->nearest_neighbors->updateSnapshots(this->snapshot_parameters, fom_solution);
        this->current_pod->addSnapshot(fom_solution);
        this->current_pod->computeBasis();

        // Update previous ROM errors with updated this->current_pod
        for(auto it = this->rom_locations.begin(); it != this->rom_locations.end(); ++it){
            it->get()->compute_initial_rom_to_final_rom_error(this->current_pod);
            it->get()->compute_total_error();
        }

        updateNearestExistingROMs(max_error_params);

        rom_points = this->nearest_neighbors->kNearestNeighborsMidpoint(max_error_params);
        this->pcout << rom_points << std::endl;

        placeROMLocations(rom_points);

        // Update max error
        max_error_params = this->getMaxErrorROM();

        this->pcout << "Max error is: " << this->max_error << std::endl;
        iteration++;
    }

    this->outputIterationData("final");

    return 0;
}

template <int dim, int nstate>
bool AdaptiveSampling<dim, nstate>::placeROMLocations(const MatrixXd& rom_points) const{
    bool error_greater_than_tolerance = false;
    for(auto midpoint : rom_points.rowwise()){

        // Check if ROM point already exists as another ROM point
        auto element = std::find_if(this->rom_locations.begin(), this->rom_locations.end(), [&midpoint](std::unique_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>& location){ return location->parameter.isApprox(midpoint);} );

        // Check if ROM point already exists as a snapshot
        bool snapshot_exists = false;
        for(auto snap_param : this->snapshot_parameters.rowwise()){
            if(snap_param.isApprox(midpoint)){
                snapshot_exists = true;
            }
        }

        if(element == this->rom_locations.end() && snapshot_exists == false){
            std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(midpoint);
            this->rom_locations.emplace_back(std::make_unique<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>(midpoint, std::move(rom_solution)));
            if(abs(this->rom_locations.back()->total_error) > this->all_parameters->reduced_order_param.adaptation_tolerance){
                error_greater_than_tolerance = true;
            }
        }
        else{
            this->pcout << "ROM already computed." << std::endl;
        }
    }
    return error_greater_than_tolerance;
}

template <int dim, int nstate>
void AdaptiveSampling<dim, nstate>::updateNearestExistingROMs(const RowVectorXd& /*parameter*/) const{

    this->pcout << "Verifying ROM points for recomputation." << std::endl;
    // Assemble ROM points in a matrix
    MatrixXd rom_points(0,0);
    for(auto it = this->rom_locations.begin(); it != this->rom_locations.end(); ++it){
        rom_points.conservativeResize(rom_points.rows()+1, it->get()->parameter.cols());
        rom_points.row(rom_points.rows()-1) = it->get()->parameter;
    }

    // Get distances between each ROM point and all other ROM points
    for(auto point : rom_points.rowwise()) {
        ProperOrthogonalDecomposition::MinMaxScaler scaler;
        MatrixXd scaled_rom_points = scaler.fit_transform(rom_points);
        RowVectorXd scaled_point = scaler.transform(point);

        VectorXd distances = (scaled_rom_points.rowwise() - scaled_point).rowwise().squaredNorm();

        std::vector<int> index(distances.size());
        std::iota(index.begin(), index.end(), 0);

        std::sort(index.begin(), index.end(),
                  [&](const int &a, const int &b) {
                      return distances[a] < distances[b];
                  });

        this->pcout << "Searching ROM points near: " << point << std::endl;
        double local_mean_error = 0;
        for (int i = 1; i < rom_points.cols() + 2; i++) {
            local_mean_error = local_mean_error + std::abs(this->rom_locations[index[i]]->total_error);
        }
        local_mean_error = local_mean_error / (rom_points.cols() + 1);
        if ((std::abs(this->rom_locations[index[0]]->total_error) > this->all_parameters->reduced_order_param.recomputation_coefficient * local_mean_error) || (std::abs(this->rom_locations[index[0]]->total_error) < (1/this->all_parameters->reduced_order_param.recomputation_coefficient) * local_mean_error)) {
            this->pcout << "Total error greater than tolerance. Recomputing ROM solution" << std::endl;
            std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(this->rom_locations[index[0]]->parameter);
            std::unique_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim, nstate>> rom_location = std::make_unique<ProperOrthogonalDecomposition::ROMTestLocation<dim, nstate>>(this->rom_locations[index[0]]->parameter, std::move(rom_solution));
            this->rom_locations[index[0]] = std::move(rom_location);
        }
    }
}

template <int dim, int nstate>
std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> AdaptiveSampling<dim, nstate>::solveSnapshotROM(const RowVectorXd& parameter) const{
    this->pcout << "Solving ROM at " << parameter << std::endl;
    Parameters::AllParameters params = this->reinitParams(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg, this->current_pod);
    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->ode_solver->steady_state();

    // Create functional
    std::shared_ptr<Functional<dim,nstate,double>> functional = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver->dg);
    functional->evaluate_functional( true, false, false);

    dealii::LinearAlgebra::distributed::Vector<double> solution(flow_solver->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> gradient(functional->dIdw);

    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> rom_solution = std::make_unique<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>>(params, solution, gradient);
    this->pcout << "Done solving ROM." << std::endl;

    return rom_solution;
}

#if PHILIP_DIM==1
        template class AdaptiveSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class AdaptiveSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}