#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/convergence_table.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "numerical_flux/numerical_flux.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"
#include "convection_diffusion_explicit_periodic.h"

#include<fenv.h>

namespace PHiLiP {
namespace Tests {
template <int dim, int nstate>
ConvectionDiffusionPeriodic<dim, nstate>::ConvectionDiffusionPeriodic(const PHiLiP::Parameters::AllParameters *const parameters_input)
:
TestsBase::TestsBase(parameters_input)
, mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}


template <int dim, int nstate>
int ConvectionDiffusionPeriodic<dim, nstate>::run_test() const
{

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  

    const unsigned int n_grids = 6;
    std::array<double,n_grids> grid_size;
    std::array<double,n_grids> soln_error;
   // std::array<double,n_grids> output_error;
    using ADtype = double;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;

#if 0
    std::vector<int> n_1d_cells(n_grids);
    n_1d_cells[0] =1;
    for (unsigned int igrid=1;igrid<n_grids;++igrid) {
        n_1d_cells[igrid] = static_cast<int>(n_1d_cells[igrid-1]*1.5) + 2;
    }
#endif
	double left = -1.0;
	double right = 1.0;
        if(dim==1 || dim == 2){
	    left = 0.0;
            const double pi = atan(1)*4.0;
	    right = 2.0 * pi;
        }
	const bool colorize = true;
//	int n_refinements = 6;
	unsigned int n_refinements = n_grids;
        dealii::ConvergenceTable convergence_table;
        std::array<FR_enum, 4> FR_arr_c;
        std::array<FR_Aux_enum, 4> FR_arr_k;
        FR_arr_c[0] = FR_enum::cDG;
        FR_arr_c[1] = FR_enum::cDG;
        FR_arr_c[2] = FR_enum::cPlus;
        FR_arr_c[3] = FR_enum::cPlus;
        FR_arr_k[0] = FR_Aux_enum::kDG;
        FR_arr_k[1] = FR_Aux_enum::kPlus;
        FR_arr_k[2] = FR_Aux_enum::kDG;
        FR_arr_k[3] = FR_Aux_enum::kPlus;
        std::vector<std::string> FR_arr_string_c;
        std::vector<std::string> FR_arr_string_k;
        FR_arr_string_c.push_back("cDG");
        FR_arr_string_c.push_back("cDG");
        FR_arr_string_c.push_back("cPlus");
        FR_arr_string_c.push_back("cPlus");
        FR_arr_string_k.push_back("kDG");
        FR_arr_string_k.push_back("kPlus");
        FR_arr_string_k.push_back("kDG");
        FR_arr_string_k.push_back("kPlus");
        const unsigned int igrid_start = 5;
//	unsigned int poly_degree = 2;
//	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);
    for(unsigned int poly_degree = 2; poly_degree<4; poly_degree++){

    for(unsigned int corr_iter=0; corr_iter<4; corr_iter++){
    
        all_parameters_new.flux_reconstruction_type = FR_arr_c[corr_iter];
        all_parameters_new.flux_reconstruction_aux_type = FR_arr_k[corr_iter];


    for(unsigned int igrid=igrid_start; igrid<n_refinements; ++igrid){
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
			dealii::Triangulation<dim> grid(
				typename dealii::Triangulation<dim>::MeshSmoothing(
					dealii::Triangulation<dim>::smoothing_on_refinement |
					dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
			dealii::parallel::distributed::Triangulation<dim> grid(
				this->mpi_communicator);
#endif

	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);

 #if PHILIP_DIM==1
#else
	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
                if(dim == 2)
		    dealii::GridTools::collect_periodic_faces(grid,2,3,1,matched_pairs);
                if(dim==3)
		    dealii::GridTools::collect_periodic_faces(grid,4,5,2,matched_pairs);
		grid.add_periodicity(matched_pairs);
#endif
	grid.refine_global(igrid);

//	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, &grid);
	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, &grid);

	double finalTime = 2.0;
    //    finalTime=0.0001;
       // finalTime=1.0;
//#if 0
        //Loop for tau test case IP stability
        unsigned int itau = 0;
        unsigned int jiteration = 0;
        double penalty =0.0;
        if(poly_degree == 2){
            penalty = 14.0;
        }
        if(poly_degree == 3){
            penalty = 29.0;
        }
        double dtau =1.0;

//itau=2;
        while (itau < 3){
//#endif

	dg->allocate_system ();

	std::cout << "Implement initial conditions" << std::endl;
	dealii::FunctionParser<dim> initial_condition;
	std::string variables;
        if (dim == 3)
	    variables = "x,y,z";
        if (dim == 2)
	    variables = "x,y";
        if (dim == 1)
	    variables = "x";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::string expression;
        if (dim == 3)
	    expression = "sin(pi*x)*sin(pi*y)*sin(pi*z)";
        if (dim == 2)
	    //expression = "sin(pi*x)*sin(pi*y)";
	    expression = "sin(x)*sin(y)";
        if(dim==1)
	    expression = "sin(x) + cos(x)";
	initial_condition.initialize(variables,
								 expression,
								 constants);
	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
	
#if 0
//if use legendre poly to interpolate IC
                const unsigned int n_quad_pts1      = dg->volume_quadrature_collection[2].size();
                const unsigned int n_dofs_cell1     =dg->fe_collection[2].dofs_per_cell;
        dealii::FullMatrix<double> Chi_operator(n_quad_pts1, n_dofs_cell1);
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell1; ++itest) {
        for (unsigned int iquad=0; iquad<n_quad_pts1; ++iquad) {
            const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[2].point(iquad);
            Chi_operator[iquad][itest] = dg->fe_collection[2].shape_value_component(itest,qpoint,istate);
        }
    }
    }
    dealii::FullMatrix<double> Chi_inv_operator(n_quad_pts1, n_dofs_cell1);
    Chi_inv_operator.invert(Chi_operator);
            const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
	
                current_dofs_indices.resize(n_dofs_cell1);
                current_cell->get_dof_indices (current_dofs_indices);
                for(unsigned int idof=0; idof<n_dofs_cell1; idof++){
                    dg->solution[current_dofs_indices[idof]]=0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts1; iquad++){
                    const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[2].point(iquad);
                       for (int idim=0; idim<dim; idim++){
                        dg->solution[current_dofs_indices[idof]] +=Chi_inv_operator[idof][iquad] *exp(-20*(qpoint[idim]-1)*(qpoint[idim]-1)); 
                        }
                    }   
                }


}
#endif
	
	
	// Create ODE solver using the factory and providing the DG object
	std::shared_ptr<PHiLiP::ODE::ODESolver<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);


        dg->penalty = penalty;
	//double dt = all_parameters->ode_solver_param.initial_time_step;
        try {
            ode_solver->advance_solution_time(finalTime);
    
//#if 0
        //get max u(x,y,t)
        double max_u = 0.0;
            const unsigned int n_quad_pts1      = dg->volume_quadrature_collection[poly_degree].size();
            const unsigned int n_dofs_cell1     =dg->fe_collection[poly_degree].dofs_per_cell;
            dealii::FullMatrix<double> Chi_operator(n_quad_pts1, n_dofs_cell1);
            for(int istate=0; istate<nstate; istate++){
                for (unsigned int itest=0; itest<n_dofs_cell1; ++itest) {
                    for (unsigned int iquad=0; iquad<n_quad_pts1; ++iquad) {
                        const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[poly_degree].point(iquad);
                        Chi_operator[iquad][itest] = dg->fe_collection[poly_degree].shape_value_component(itest,qpoint,istate);
                    }
                }
            }
            const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
	
                std::vector<double> soln_at_q_init(n_quad_pts1);
                current_dofs_indices.resize(n_dofs_cell1);
                current_cell->get_dof_indices (current_dofs_indices);
                for(unsigned int iquad=0; iquad<n_quad_pts1; iquad++){
                    soln_at_q_init[iquad] = 0.0;
                    for(unsigned int idof=0; idof<n_dofs_cell1; idof++){
                        soln_at_q_init[iquad] +=Chi_operator[iquad][idof] * dg->solution[current_dofs_indices[idof]];
                    }   
                    if (soln_at_q_init[iquad] > max_u)
                        max_u = soln_at_q_init[iquad];
                }
            }
    
        printf("max u %g, penalty %g iteration %d i iteration %d\n",max_u, penalty, jiteration, itau);
        fflush(stdout);
        double abs_max_u = dealii::Utilities::MPI::max(max_u,mpi_communicator);
        printf("abs max u %g\n",abs_max_u);
        fflush(stdout);
        std::cout << "for c: " << FR_arr_string_c[corr_iter]
                << " , for k: " << FR_arr_string_k[corr_iter]
                 << std::endl;
        if(abs_max_u > 2.0){//unstable
          //  dg->penalty += dtau;
            penalty += dtau;
            jiteration++;
            if (poly_degree == 2 && penalty >16.0){
               // all_parameters->ode_solver_param.initial_time_step /= 10.0;
                all_parameters_new.ode_solver_param.initial_time_step /= 10.0;
                penalty = 13.0;
                jiteration--;
            }
            if (poly_degree == 3 && penalty >31.0){
               // all_parameters->ode_solver_param.initial_time_step /= 10.0;
                all_parameters_new.ode_solver_param.initial_time_step /= 10.0;
                penalty = 28.0;
                jiteration--;
            }
        
        }
        else{//it's stable
         //   dg->penalty -= dtau;
            penalty -= dtau;
            dtau /=10.0;
            itau++;
            if (jiteration == 0){
                dtau *= 10.0;
                itau--;
            }
        }

        } catch(int e) {//unstable
            std::cout << "caught error " << std::endl;
            penalty += dtau;
            jiteration++;
        printf("penalty %g iteration %d i iteration %d\n",penalty, jiteration, itau);
        fflush(stdout);
            if (poly_degree == 2 && penalty >16.0){
               // all_parameters->ode_solver_param.initial_time_step /= 10.0;
                all_parameters_new.ode_solver_param.initial_time_step /= 10.0;
            }
            if (poly_degree == 3 && penalty >31.0){
               // all_parameters->ode_solver_param.initial_time_step /= 10.0;
                all_parameters_new.ode_solver_param.initial_time_step /= 10.0;
            }
        } catch( ... ) {//unstable
            std::cout << "caught error " << std::endl;
            penalty += dtau;
            jiteration++;
            if (poly_degree == 2 && penalty >16.0){
               // all_parameters->ode_solver_param.initial_time_step /= 10.0;
                all_parameters_new.ode_solver_param.initial_time_step /= 10.0;
            }
            if (poly_degree == 3 && penalty >31.0){
               // all_parameters->ode_solver_param.initial_time_step /= 10.0;
                all_parameters_new.ode_solver_param.initial_time_step /= 10.0;
            }
        }
        }//end of tau loop

        printf(" iterations %d with penalty %g for poly %d with time step %g\n",jiteration, dg->penalty,
        poly_degree, all_parameters_new.ode_solver_param.initial_time_step);
       // poly_degree, all_parameters->ode_solver_param.initial_time_step);
        fflush(stdout); 
        std::cout << "for c: " << FR_arr_string_c[corr_iter]
                << " , for k: " << FR_arr_string_k[corr_iter]
                 << std::endl;
//#endif

//output results
            const unsigned int n_global_active_cells = grid.n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim
                 << "\t Polynomial degree p: " << poly_degree
                 << std::endl
                 << "Grid number: " << igrid << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            //dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            //dealii::MappingQ<dim,dim> mappingq(dg->max_degree+1);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid.mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0.0;

            // Integrate solution error and output error

           // const double pi = atan(1)*4.0;
            //const double dif_coef = 0.1*pi/exp(1);
           // const double dif_coef = 0.1;
            const double dif_coef = 0.5;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    for (int istate=0; istate<nstate; ++istate) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact=1.0;
                    for(int idim=0; idim<dim; idim++){
                       // uexact *= sin(pi*qpoint[idim]);;
                        uexact *= sin(qpoint[idim]);;
                    }
                       // uexact *= exp(- 2 * dif_coef * pow(pi,dim) * finalTime);;
                        uexact *= exp(- 2 * dif_coef * finalTime);;
                        if(dim==1)
                            uexact=exp(-dif_coef*finalTime)*(sin(qpoint[0])+cos(qpoint[0]));
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                    }
                }

            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));


            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
     //       output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

//#if 0
            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("c correction", FR_arr_string_c[corr_iter]);
            convergence_table.add_value("k correction", FR_arr_string_k[corr_iter]);
            convergence_table.add_value("Tau Penalty", penalty);
            convergence_table.add_value("jIterations", jiteration);
           // convergence_table.add_value("Time step", all_parameters->ode_solver_param.initial_time_step);
            convergence_table.add_value("Time step", all_parameters_new.ode_solver_param.initial_time_step);
 //           convergence_table.add_value("output_error", output_error[igrid]);
//#endif


            pcout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                // << " Residual: " << ode_solver->residual_norm
                 << std::endl;

#if 0
            pcout //<< " output_exact: " << exact_solution_integral
                 //<< " output_discrete: " << solution_integral
                 << " output_error: " << output_error[igrid]
                 << std::endl;
#endif

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
              //  const double slope_output_err = log(output_error[igrid]/output_error[igrid-1])
     //                                 / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << soln_error[igrid-1]
                     << "  solution_error2 " << soln_error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
#if 0
                     << "  solution_integral_error1 " << output_error[igrid-1]
                     << "  solution_integral_error2 " << output_error[igrid]
                     << "  slope " << slope_output_err
                     << std::endl;
#endif
            }

    }//end of grid loop

    }//end of correction iteration loop
    
        pcout << " ********************************************"
             << std::endl
             << " Results for Tau min for dif poly/c/k " 
             << std::endl
             << " ********************************************"
             << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
            convergence_table.set_scientific("Tau Penalty", true);
            convergence_table.set_scientific("jIterations", true);
            convergence_table.set_scientific("Time step",true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
    if( n_refinements > igrid_start +1){
        pcout << " ********************************************"
             << std::endl
             << " Convergence rates for p = " << poly_degree
             << std::endl
             << " ********************************************"
             << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("output_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
    }

    }//end of poly_degree loop

       // convergence_table_vector.push_back(convergence_table);

	return 0; //need to change
}

//int main (int argc, char * argv[])
//{
//	//parse parameters first
//	feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
//	dealii::deallog.depth_console(99);
//		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
//		const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
//		const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
//		dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
//		pcout << "Starting program with " << n_mpi << " processors..." << std::endl;
//		if ((PHILIP_DIM==1) && !(n_mpi==1)) {
//			std::cout << "********************************************************" << std::endl;
//			std::cout << "Can't use mpirun -np X, where X>1, for 1D." << std::endl
//					  << "Currently using " << n_mpi << " processors." << std::endl
//					  << "Aborting..." << std::endl;
//			std::cout << "********************************************************" << std::endl;
//			std::abort();
//		}
//	int test_error = 1;
//	try
//	{
//		// Declare possible inputs
//		dealii::ParameterHandler parameter_handler;
//		PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
//		PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);
//
//		// Read inputs from parameter file and set those values in AllParameters object
//		PHiLiP::Parameters::AllParameters all_parameters;
//		std::cout << "Reading input..." << std::endl;
//		all_parameters.parse_parameters (parameter_handler);
//
//		AssertDimension(all_parameters.dimension, PHILIP_DIM);
//
//		std::cout << "Starting program..." << std::endl;
//
//		using namespace PHiLiP;
//		//const Parameters::AllParameters parameters_input;
//		AdvectionPeriodic<PHILIP_DIM, 1> advection_test(&all_parameters);
//		int i = advection_test.run_test();
//		return i;
//	}
//	catch (std::exception &exc)
//	{
//		std::cerr << std::endl << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Exception on processing: " << std::endl
//				  << exc.what() << std::endl
//				  << "Aborting!" << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl;
//		return 1;
//	}
//
//	catch (...)
//	{
//		std::cerr << std::endl
//				  << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Unknown exception!" << std::endl
//				  << "Aborting!" << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl;
//		return 1;
//	}
//	std::cout << "End of program." << std::endl;
//	return test_error;
//}

    template class ConvectionDiffusionPeriodic <PHILIP_DIM,1>;

} //Tests namespace
} //PHiLiP namespace







