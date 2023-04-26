#include <fstream>
#include "dg/dg_factory.hpp"
#include "TGV_scaling.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"
#include <time.h>
#include <deal.II/base/timer.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreenScaling<dim, nstate>::EulerTaylorGreenScaling(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}
template<int dim, int nstate>
std::array<double,2> EulerTaylorGreenScaling<dim, nstate>::compute_change_in_entropy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    dealii::LinearAlgebra::distributed::Vector<double> energy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        std::array<std::vector<double>,nstate> energy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = euler_double->compute_entropy_variables(soln_state);
            std::array<double,nstate> kin_energy_state = euler_double->compute_kinetic_energy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
                    energy_var_at_q[istate].resize(n_quad_pts);
                }
                energy_var_at_q[istate][iquad] = kin_energy_state[istate];
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
            std::vector<double> energy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(energy_var_at_q[istate], energy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
                energy_var_hat_global[dofs_indices[idof]] = energy_var_hat[ishape];
            }
        }
    }

    dg->assemble_residual();
    std::array<double,2> change_entropy_and_energy;
    change_entropy_and_energy[0] = entropy_var_hat_global * dg->right_hand_side;
    change_entropy_and_energy[1] = energy_var_hat_global * dg->right_hand_side;
    return change_entropy_and_energy;
}

template<int dim, int nstate>
double EulerTaylorGreenScaling<dim, nstate>::compute_entropy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the entropy evaluated in the broken Sobolev-norm rather than L2-norm
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(dg->all_parameters->use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = pde_physics_double->compute_entropy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
        }
    }

    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    return entropy;
}

template<int dim, int nstate>
double EulerTaylorGreenScaling<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the energy in the L2-norm (physically relevant)
    int overintegrate = 10 ;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    double total_kinetic_energy = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        //Please see Eq. 3.21 in Gassner, Gregor J., Andrew R. Winters, and David A. Kopriva. "Split form nodal discontinuous Galerkin schemes with summation-by-parts property for the compressible Euler equations." Journal of Computational Physics 327 (2016): 39-66.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
             const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            const double density = soln_at_q[0];

            const double quadrature_kinetic_energy =  0.5*(soln_at_q[1]*soln_at_q[1]
                                                    + soln_at_q[2]*soln_at_q[2]
                                                    + soln_at_q[3]*soln_at_q[3])/density;

            total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
        }
    }
    return total_kinetic_energy;
}

template<int dim, int nstate>
double EulerTaylorGreenScaling<dim, nstate>::get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
    //get local CFL
    const unsigned int n_dofs_cell = nstate*pow(poly_degree+1,dim);
    const unsigned int n_quad_pts = pow(poly_degree+1,dim);
    std::vector<dealii::types::global_dof_index> dofs_indices1 (n_dofs_cell);

    double cfl_min = 1e100;
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices1);
        std::vector< std::array<double,nstate>> soln_at_q(n_quad_pts);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int istate=0; istate<nstate; istate++) {
                soln_at_q[iquad][istate]      = 0;
            }
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
          dealii::Point<dim> qpoint = dg->volume_quadrature_collection[poly_degree].point(iquad);
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                soln_at_q[iquad][istate] += dg->solution[dofs_indices1[idof]] * dg->fe_collection[poly_degree].shape_value_component(idof, qpoint, istate);
            }
        }

        std::vector< double > convective_eigenvalues(n_quad_pts);
        for (unsigned int isol = 0; isol < n_quad_pts; ++isol) {
            convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_at_q[isol]);
        }
        const double max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));

        const double max_eig_mpi = dealii::Utilities::MPI::max(max_eig, mpi_communicator);
        double cfl = 0.1 * delta_x/max_eig_mpi;
        if(cfl < cfl_min)
            cfl_min = cfl;

    }
    return cfl_min;
}

template <int dim, int nstate>
int EulerTaylorGreenScaling<dim, nstate>::run_test() const
{
    // using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    // std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (mpi_communicator);
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = 0.0;
    double right = 2 * dealii::numbers::PI;

    const unsigned int n_refinements = all_parameters->flow_solver_param.number_of_grid_elements_per_dimension;
    if(all_parameters->use_curvilinear_grid){
        //if curvilinear
        PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);
    }
    else{
        //if straight
        const bool colorize = true;
        dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
        grid->add_periodicity(matched_pairs);
        grid->refine_global(n_refinements);
    }

    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);
    const unsigned int poly_degree_start= all_parameters->flow_solver_param.poly_degree;
   // const unsigned int poly_degree_start= 11;

   // const unsigned int poly_degree_end = 20;//poly_degree=10 is as high can go locally
    const unsigned int poly_degree_end = 25;//poly_degree=10 is as high can go locally
 //   const unsigned int poly_degree_end = 12;//poly_degree=10 is as high can go locally
  //  const unsigned int poly_degree_end = 10;//poly_degree=10 is as high can go locally
  //  const unsigned int poly_degree_end = 8;//poly_degree=10 is as high can go locally
    //const unsigned int poly_degree_end = all_parameters->flow_solver_param.max_poly_degree_for_adaptation;//poly_degree=10 is as high can go locally
  //  std::array<clock_t,poly_degree_end> time_to_run;
  //  std::array<clock_t,poly_degree_end> time_to_run_mpi;
    std::array<double,poly_degree_end> time_to_run;
    std::array<double,poly_degree_end> time_to_run_mpi;

    for(unsigned int poly_degree = poly_degree_start; poly_degree<poly_degree_end; poly_degree++){

        // set the warped grid
       // const unsigned int grid_degree = poly_degree;
     //   const unsigned int grid_degree = 1;
        const unsigned int grid_degree = (all_parameters->use_curvilinear_grid) ? poly_degree : 1;
         
//        const unsigned int grid_degree = 1;
        if(all_parameters->overintegration == 100){
            if(all_parameters->use_curvilinear_grid){
                all_parameters_new.overintegration = 2*(poly_degree+1);
            }
            else{
                all_parameters_new.overintegration = poly_degree+1;
            }
        }
         
        // Create DG
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        dg->allocate_system (false,false,false);
         
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Implement initial conditions" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                    InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
        std::cout << "Implemented initial conditions" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
         
 //       const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
 //       double delta_x = (right-left)/pow(n_global_active_cells2,1.0/dim)/(poly_degree+1.0);
 //       pcout<<" delta x "<<delta_x<<std::endl;
         
 //       all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
 //       std::cout << "got timestep, about to create ode solver" << std::endl;
 //       MPI_Barrier(MPI_COMM_WORLD);
         

//        std::cout << "creating ODE solver" << std::endl;
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "ODE solver successfully created" << std::endl;
#if 0
       // const double finalTime = all_parameters_new.flow_solver_param.final_time;
       const double time_step_start = get_timestep(dg,poly_degree,delta_x);
        const double dt_start = dealii::Utilities::MPI::min(time_step_start, mpi_communicator);
       // const double finalTime = dt_start;
      //  const double finalTime = 100.0 * dt_start;
       // const double finalTime = 10000.0 * dt_start;
        const double finalTime = 10.0 * dt_start;
         
        std::cout << " number dofs " << dg->dof_handler.n_dofs()<<std::endl;
        std::cout << "preparing to advance solution in time" << std::endl;
#endif
         
         
        ode_solver->current_iteration = 0;
        ode_solver->allocate_ode_system();
    dealii::LinearAlgebra::distributed::Vector<double> solution_update;
    solution_update.reinit(dg->locally_owned_dofs, dg->ghost_dofs, this->mpi_communicator);

      //  clock_t time_start;
      //  time_start = clock();
       // while(ode_solver->current_time < finalTime){
        for(unsigned int i_step=0; i_step<10; i_step++){
//            const double time_step =  get_timestep(dg,poly_degree, delta_x);
 //           if(ode_solver->current_iteration%all_parameters_new.ode_solver_param.print_iteration_modulo==0)
 //               pcout<<"time step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
//            const double dt = dealii::Utilities::MPI::min(time_step, mpi_communicator);
//            ode_solver->step_in_time(dt, false);
            dg->assemble_residual();
            //me apply in loo
            if(all_parameters->use_inverse_mass_on_the_fly){
                dg->apply_inverse_global_mass_matrix(dg->right_hand_side, solution_update); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            } else{
                dg->global_inverse_mass_matrix.vmult(solution_update, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            }
           // ode_solver->advance_solution_time(dt);
      //      ode_solver->current_iteration += 1;
 //           const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
 //           if (is_output_iteration) {
 //               const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
 //               dg->output_results_vtk(file_number);
 //           }
       //     const unsigned int n_quad= pow(poly_degree+1,dim);
       //     dealii::FullMatrix<double> wrong_scaling(n_quad,n_quad);
       //     for(unsigned int i=0;i<n_quad;i++){
       //     for(unsigned int j=0;j<n_quad;j++){
       //         wrong_scaling[i][j]=1.0;
       //     }
       //     }
        }

       // time_to_run[poly_degree] = clock() - time_start;
        time_to_run[poly_degree] = dg->assemble_residual_time;
       // time_to_run_mpi[poly_degree] = dealii::Utilities::MPI::sum(time_to_run[poly_degree], mpi_communicator);
//        time_to_run_mpi[poly_degree] = time_to_run[poly_degree];
        time_to_run_mpi[poly_degree] = dealii::Utilities::MPI::sum(time_to_run[poly_degree], this->mpi_communicator);
    //    pcout<<"Poly Degree "<<poly_degree<<" time to run Mpi "<<std::fixed << std::setprecision(16) << (double)time_to_run_mpi[poly_degree]/CLOCKS_PER_SEC<<std::endl;
        pcout<<"Poly Degree "<<poly_degree<<" time to run Mpi "<<std::fixed << std::setprecision(16) << (double)time_to_run_mpi[poly_degree]<<std::endl;
       // myfile << poly_degree << " " << std::fixed << std::setprecision(16) << (double)time_to_run_mpi[poly_degree]/CLOCKS_PER_SEC << std::endl;
        myfile << poly_degree << " " << std::fixed << std::setprecision(16) << time_to_run_mpi[poly_degree]<< std::endl;
    }//end of poly loop


    myfile.close();
    double avg_slope1 = 0.0;
    pcout<<"Times for one timestep"<<std::endl;
    pcout<<"local time  | Slope |  "<<"MPI sum time | Slope "<<std::endl;
    for(unsigned int i=poly_degree_start+1; i<poly_degree_end; i++){
       // pcout<<(double)time_to_run[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_to_run[i]/CLOCKS_PER_SEC) / ((double)time_to_run[i-1]/CLOCKS_PER_SEC))
        pcout<<(double)time_to_run[i]<<" "<<std::log(((double)time_to_run[i]) / ((double)time_to_run[i-1]))
                       // / std::log((double)((i)/(i-1.0)))<<" "<<
                        / std::log((double)((i+1.0)/(i)))<<" "<<
        (double)time_to_run_mpi[i]<<" "<<std::log(((double)time_to_run_mpi[i]) /( (double)time_to_run_mpi[i-1]))
                       // / std::log((double)((i)/(i-1.0)))<<
                        / std::log((double)((i+1.0)/(i)))<<
        std::endl;
        if(i>poly_degree_end-4){
            avg_slope1 += std::log(((double)time_to_run[i]) /( (double)time_to_run_mpi[i-1]))
                       // / std::log((double)((i)/(i-1.0)));
                        / std::log((double)((i+1.0)/(i)));
        }
    }
    avg_slope1 /= (3.0);


    return 0;
}

#if PHILIP_DIM==3
    template class EulerTaylorGreenScaling <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

