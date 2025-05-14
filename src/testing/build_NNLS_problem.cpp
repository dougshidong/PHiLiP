#include "build_NNLS_problem.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/assemble_ECSW_residual.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include "reduced_order/pod_adaptive_sampling.h"
#include "reduced_order/adaptive_sampling_base.h"
#include <iostream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BuildNNLSProblem<dim, nstate>::BuildNNLSProblem(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

std::shared_ptr<Epetra_CrsMatrix> local_generate_test_basis(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis){
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver){ 
        return std::make_shared<Epetra_CrsMatrix>(pod_basis);
    }
    else if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver){ 
        Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
        Epetra_CrsMatrix petrov_galerkin_basis(Epetra_DataAccess::Copy, system_matrix_rowmap, pod_basis.NumGlobalCols());
        EpetraExt::MatrixMatrix::Multiply(system_matrix, false, pod_basis, false, petrov_galerkin_basis, true);

        return std::make_shared<Epetra_CrsMatrix>(petrov_galerkin_basis);
    }
    else {
        return nullptr;
    }
}


template <int dim, int nstate>
int BuildNNLSProblem<dim, nstate>::run_test() const
{
    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    // Create flow solver and adaptive sampling class instances
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);

    // Place minimum number of snapshots in the parameter space (3 snapshots in 1 parameter cases)
    parameter_sampling->configureInitialParameterSpace();
    parameter_sampling->placeInitialSnapshots();
    parameter_sampling->current_pod->computeBasis();
    MatrixXd snapshot_parameters = parameter_sampling->snapshot_parameters;

    // Create instance of NNLS Problem assembler
    std::cout << "Construct instance of Assembler..."<< std::endl;
    HyperReduction::AssembleECSWRes<dim,nstate> constructor_NNLS_problem(all_parameters, parameter_handler, flow_solver_petrov_galerkin->dg, parameter_sampling->current_pod, snapshot_parameters, ode_solver_type, Comm);
    
    // Add in FOM snapshots from sampling
    constructor_NNLS_problem.fom_locations = parameter_sampling->fom_locations;
    
    std::cout << "Build Problem..."<< std::endl;
    constructor_NNLS_problem.build_problem();

    /* UNCOMMENT TO SAVE THE RESIDUAL AND TEST BASIS FOR EACH OF THE SNAPSHOTS, used to feed MATLAB and build C/d
    std::shared_ptr<DGBase<dim,double>> dg = flow_solver_petrov_galerkin->dg;
    MatrixXd snapshotMatrix = parameter_sampling->current_pod->getSnapshotMatrix();
    const Epetra_CrsMatrix epetra_pod_basis = parameter_sampling->current_pod->getPODBasis()->trilinos_matrix();
    Epetra_CrsMatrix epetra_system_matrix = dg->system_matrix.trilinos_matrix();

    int N_e = dg->triangulation->n_active_cells(); // Number of elements (? should be the same as N ?)
    int snap_num = 0;
    for(auto snap_param : parameter_sampling->snapshot_parameters.rowwise()){
        std::cout << "Extract Snapshot from matrix"<< std::endl;
        dealii::LinearAlgebra::ReadWriteVector<double> snapshot_s;
        snapshot_s.reinit(N_e);
        for (int snap_row = 0; snap_row < N_e; snap_row++){
            snapshot_s(snap_row) = snapshotMatrix(snap_row, snap_num);
        }
        dealii::LinearAlgebra::distributed::Vector<double> reference_solution(dg->solution);
        reference_solution.import(snapshot_s, dealii::VectorOperation::values::insert);
        
        Parameters::AllParameters params = parameter_sampling->reinit_params(snap_param);
        
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        dg = flow_solver->dg;

        std::cout << "Set dg solution to snapshot"<< std::endl;
        dg->solution = reference_solution;
        // reference_solution.print(std::cout, 7);
        const bool compute_dRdW = true;
        std::cout << "Re-compute the residual"<< std::endl;
        dg->assemble_residual(compute_dRdW);

        std::cout << "Compute test basis with system matrix and pod basis"<< std::endl;
        epetra_system_matrix = dg->system_matrix.trilinos_matrix();
        std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = local_generate_test_basis(ode_solver_type, epetra_system_matrix, epetra_pod_basis);


        std::cout << "Place residual in Epetra vector"<< std::endl;
        // std::cout << "Residual"<< std::endl;
        // dg->right_hand_side.print(std::cout);
        Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::Copy, epetra_system_matrix.RowMap(), dg->right_hand_side.begin());


        snap_num +=1;

        std::ofstream out_file(std::to_string(snap_num) + "_residual.txt");
        for(int i = 0 ; i < epetra_right_hand_side.GlobalLength() ; i++){
            out_file << " " << std::setprecision(17) << epetra_right_hand_side[i] << " \n";
        }
        out_file.close();

        dealii::LAPACKFullMatrix<double> test_basis;
        test_basis.reinit(epetra_test_basis->NumGlobalRows(), epetra_test_basis->NumGlobalCols());
        for (int m = 0; m < epetra_test_basis->NumGlobalRows(); m++) {
            double *row = (*epetra_test_basis)[m];
            for (int n = 0; n < epetra_test_basis->NumGlobalCols(); n++) {
                test_basis.set(m, n, row[n]);
            }
        }
        std::ofstream basis_out_file(std::to_string(snap_num) + "_test_basis.txt");
        unsigned int precision = 16;
        test_basis.print_formatted(basis_out_file, precision);
    }
        
    */
    std::cout << "Load Matlab Results" << std::endl;
    Eigen::MatrixXd C_MAT = load_csv<MatrixXd>("C.csv");
    Eigen::MatrixXd d_MAT = load_csv<MatrixXd>("d.csv");
    Eigen::MatrixXd x_MAT = load_csv<MatrixXd>("x.csv");

    const int rank = Comm.MyPID();
    int rows = (constructor_NNLS_problem.A_T->trilinos_matrix()).NumGlobalCols();
    Epetra_Map bMap(rows, (rank == 0) ? rows: 0, 0, Comm);
    Epetra_Vector b_Epetra(bMap);
    auto b = constructor_NNLS_problem.b;
    unsigned int local_length = bMap.NumMyElements();
    for(unsigned int i = 0 ; i < local_length ; i++){
        b_Epetra[i] = b(i);
    }

    // Build NNLS Solver with C and d
    // Solver parameters
    std::cout << "Create NNLS problem..."<< std::endl;
    std::cout << all_parameters->hyper_reduction_param.NNLS_tol << std::endl;
    std::cout << all_parameters->hyper_reduction_param.NNLS_max_iter << std::endl;
    NNLSSolver NNLS_prob(all_parameters, parameter_handler, constructor_NNLS_problem.A_T->trilinos_matrix(), true, Comm, b_Epetra);
    std::cout << "Solve NNLS problem..."<< std::endl;
    // Solve NNLS problem (should return 1 if solver achieves the accuracy tau before the max number of iterations)
    bool exit_con = NNLS_prob.solve();
    
    // Extract the weights
    Epetra_Vector weights = NNLS_prob.get_solution();
    Eigen::MatrixXd weights_eig(weights.GlobalLength(),1);
    epetra_to_eig_vec(weights.GlobalLength(), weights , weights_eig);

    // Compare with the MATLAB results (expected to be close but not identical)
    exit_con &= x_MAT.isApprox(weights_eig, 1E-2);

    std::cout << "ECSW Weights"<< std::endl;
    std::cout << weights << std::endl;

    return !exit_con;
}

#if PHILIP_DIM==1
        template class BuildNNLSProblem<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class BuildNNLSProblem<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
