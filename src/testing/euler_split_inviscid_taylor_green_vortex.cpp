#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_split_inviscid_taylor_green_vortex.h"
#include "physics/initial_conditions/initial_condition.h"

#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreen<dim, nstate>::EulerTaylorGreen(const Parameters::AllParameters *const parameters_input)
:
TestsBase::TestsBase(parameters_input)
{}


template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_MK_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the energy evaluated in the broken Sobolev-norm rather than L2-norm
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 0 ;//10;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    double total_kinetic_energy = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        const unsigned int n_dofs_cell = fe_values_extra.dofs_per_cell;
        std::vector<double> Mu(n_dofs_cell);//ESFR mass matrix times solution
        for(unsigned int itest=0; itest<n_dofs_cell; itest++){
            Mu[itest] = 0.0;
            const unsigned int istate_test = fe_values_extra.get_fe().system_to_component_index(itest).first;
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                if(istate == istate_test && istate > 0 && istate < 4){
                    Mu[itest] += dg->global_mass_matrix(dofs_indices[itest],dofs_indices[idof]) * dg->solution[dofs_indices[idof]]; 
                }
            }
        }
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
            const unsigned int ishape = fe_values_extra.get_fe().system_to_component_index(idof).second;
            if(istate > 0 && istate < 4){
                total_kinetic_energy += 0.5 * Mu[idof] * dg->solution[dofs_indices[idof]] / dg->solution[dofs_indices[ishape]]; 
            }
        }
    
    }
    return total_kinetic_energy;
}
template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
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
                                                    + soln_at_q[3]*soln_at_q[3] )/density;

            total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
        }
    }
    return total_kinetic_energy;
}

template <int dim, int nstate>
int EulerTaylorGreen<dim, nstate>::run_test() const
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
//    const bool colorize = true;
    const int n_refinements = 2;
    unsigned int poly_degree = 3;
    const unsigned int grid_degree = poly_degree;

    //set the warped grid
    PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);


//    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
     
//    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
//    dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
//    dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
//    dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
//    grid->add_periodicity(matched_pairs);
//     
//    grid->refine_global(n_refinements);

    //Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    std::cout << "Implement initial conditions" << std::endl;
    InitialCondition<dim,nstate,double> initial_condition(dg, &all_parameters_new);

    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double delta_x = (right-left)/n_global_active_cells2/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    //get local CFL
    const unsigned int n_dofs_cell = nstate*pow(poly_degree+1,dim);
    const unsigned int n_quad_pts = pow(poly_degree+1,dim);
    std::vector<dealii::types::global_dof_index> dofs_indices1 (n_dofs_cell);

    double cfl_min = 1e100;
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&all_parameters_new);
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
        double cfl = 0.1 * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;

    }
    all_parameters_new.ode_solver_param.initial_time_step =  cfl_min;
    pcout<<"The new timestep with eigenval is "<<cfl_min<<" compared to previous "<<0.1 * delta_x<<std::endl;
     
    std::cout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout << "ODE solver successfully created" << std::endl;
    double finalTime = 14.;
//    finalTime = 0.1;//to speed things up locally in tests, doesn't need full 14seconds to verify.
    double dt = all_parameters_new.ode_solver_param.initial_time_step;
//    double dt = all_parameters_new.ode_solver_param.initial_time_step / 10.0;
 

    finalTime = dt;

    std::cout<<" number dofs "<<
            dg->dof_handler.n_dofs()<<std::endl;
    std::cout << "preparing to advance solution in time" << std::endl;

    //currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
    //this causes some issues with outputs (only one file is output, which is overwritten at each time step)
    //also the ode solver output doesn't make sense (says "iteration 1 out of 1")
    //but it works. I'll keep it for now and need to modify the output functions later to account for this.
    double initialcond_energy = compute_kinetic_energy(dg, poly_degree);
    double initialcond_energy_mpi = (dealii::Utilities::MPI::sum(initialcond_energy, mpi_communicator));
    std::cout << std::setprecision(16) << std::fixed;
    pcout << "Energy for initial condition " << initialcond_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;

    pcout << "Energy at time " << 0 << " is " << compute_kinetic_energy(dg, poly_degree) << std::endl;
    ode_solver->current_iteration = 0;
	ode_solver->advance_solution_time(dt/10.0);
	double initial_energy = compute_kinetic_energy(dg, poly_degree);
	double initial_energy_mpi = (dealii::Utilities::MPI::sum(initial_energy, mpi_communicator));
        double initial_MK_energy = compute_MK_energy(dg, poly_degree);
    
    std::cout << std::setprecision(16) << std::fixed;
    pcout << "Energy at one timestep is " << initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
   // std::ofstream myfile ("kinetic_energy_3D_TGV_cdg_curv_grid_4x4.gpl" , std::ios::trunc);
    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);

    for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
    {
        ode_solver->advance_solution_time(dt);
       // double current_energy = compute_kinetic_energy(dg,poly_degree) / initial_energy;
        double current_energy = compute_kinetic_energy(dg,poly_degree);
        double current_energy_mpi = (dealii::Utilities::MPI::sum(current_energy, mpi_communicator))/initial_energy_mpi;
        std::cout << std::setprecision(16) << std::fixed;
       // pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
        pcout << "Energy at time " << i * dt << " is " << current_energy_mpi << std::endl;
        pcout << "Actual Energy Divided by volume at time " << i * dt << " is " << current_energy_mpi*initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
       // myfile << i * dt << " " << current_energy << std::endl;
        myfile << i * dt << " " << current_energy_mpi << std::endl;
       // if (current_energy*initial_energy - initial_energy >= 1.00)
        if (current_energy_mpi*initial_energy_mpi - initial_energy_mpi >= 1.00)
          {
              pcout << " Energy was not monotonically decreasing" << std::endl;
              return 1;
          }
          double current_MK_energy = compute_MK_energy(dg, poly_degree)/initial_MK_energy;
          std::cout << std::setprecision(16) << std::fixed;
          pcout << "M plus K norm at time " << i * dt << " is " << current_MK_energy<< std::endl;
          myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_MK_energy << std::endl;
    }

    myfile.close();

    return 0;}

#if PHILIP_DIM==3
    template class EulerTaylorGreen <PHILIP_DIM,PHILIP_DIM+2>;
#endif

}
}




