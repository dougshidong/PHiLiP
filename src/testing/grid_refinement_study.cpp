#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <chrono>

#include <type_traits>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "tests.h"
#include "grid_refinement_study.h"

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"

#include "dg/dg.h"
#include "dg/dg_factory.hpp"

#include "ode_solver/ode_solver_factory.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "grid_refinement/grid_refinement.h"
#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/msh_out.h"
#include "grid_refinement/size_field.h"
#include "grid_refinement/gnu_out.h"

namespace PHiLiP {
    
namespace Tests {

template <int dim, int nstate>
class InitialConditions : public dealii::Function<dim>
{
public:
    InitialConditions() : dealii::Function<dim,double>(nstate){}

    double value(const dealii::Point<dim> &/* point */, const unsigned int /* istate */) const override
    {
        return 0.0;
    }
};

template <int dim, int nstate, typename MeshType>
GridRefinementStudy<dim,nstate,MeshType>::GridRefinementStudy(
    const Parameters::AllParameters *const parameters_input) :
        TestsBase::TestsBase(parameters_input){}

template <int dim, int nstate, typename MeshType>
int GridRefinementStudy<dim,nstate,MeshType>::run_test() const
{
    pcout << " Running Grid Refinement Study. " << std::endl;
    const Parameters::AllParameters param                = *(TestsBase::all_parameters);
    const Parameters::GridRefinementStudyParam grs_param = param.grid_refinement_study_param;

    using ADtype = Sacado::Fad::DFad<double>;

    const unsigned int poly_degree      = grs_param.poly_degree;
    const unsigned int poly_degree_max  = grs_param.poly_degree_max;
    const unsigned int poly_degree_grid = grs_param.poly_degree_grid;

    const unsigned int num_refinements = grs_param.num_refinements;

    // creating the physics object
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double
        = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param);
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,ADtype> > physics_adtype
        = Physics::PhysicsFactory<dim,nstate,ADtype>::create_Physics(&param);

    // for each of the runs, a seperate refinement table
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    // output for convergence figure
    PHiLiP::GridRefinement::GnuFig<double> gf_solution;
    PHiLiP::GridRefinement::GnuFig<double> gf_functional;

    std::vector<double> error_per_cell;

    // start of loop for each grid refinement run
    for(unsigned int iref = 0; iref <  (num_refinements?num_refinements:1); ++iref){
        // getting the parameters for this run
        const Parameters::GridRefinementParam gr_param = grs_param.grid_refinement_param_vector[iref];
        
        const unsigned int refinement_steps = gr_param.refinement_steps;

        std::shared_ptr<MeshType> grid = 
            MeshFactory<MeshType>::create_MeshType(this->mpi_communicator);

        // getting the grid from GridEnum type
        get_grid(grid, grs_param);
        
        // approximating the exact functional solution using a fine grid version
        double functional_value_exact;
        if(grs_param.approximate_functional){
            functional_value_exact = approximate_exact_functional(
                physics_double,
                physics_adtype,
                param,
                grs_param);
        }else{
            functional_value_exact = grs_param.functional_value;
        }
        std::cout << "Functional estimate = " << functional_value_exact << std::endl;

        // generate DG
        std::shared_ptr< DGBase<dim, double, MeshType> > dg 
            = DGFactory<dim,double,MeshType>::create_discontinuous_galerkin(
                &param, 
                poly_degree,
                poly_degree_max,
                poly_degree_grid,
                grid);
        dg->allocate_system();

        // initialize the solution
        std::shared_ptr<dealii::Function<dim>> initial_conditions = 
            std::make_shared<InitialConditions<dim,nstate>>();
        
        dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
        solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
        const auto mapping = *(dg->high_order_grid->mapping_fe_field);
        dealii::VectorTools::interpolate(mapping, dg->dof_handler, *initial_conditions, solution_no_ghost);
        dg->solution = solution_no_ghost;

        // generate ODE solver
        std::shared_ptr< ODE::ODESolverBase<dim,double,MeshType> > ode_solver
            = ODE::ODESolverFactory<dim,double,MeshType>::create_ODESolver(dg);
        // ode_solver->steady_state();
        // ode_solver->initialize_steady_polynomial_ramping(poly_degree);

        // generate Functional
        std::shared_ptr< Functional<dim,nstate,double,MeshType> > functional 
            = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(grs_param.functional_param, dg);

        // generate Adjoint
        std::shared_ptr< Adjoint<dim,nstate,double,MeshType> > adjoint 
            = std::make_shared< Adjoint<dim,nstate,double,MeshType> >(dg, functional, physics_adtype);

        // generate the GridRefinement
        std::shared_ptr< GridRefinement::GridRefinementBase<dim,nstate,double,MeshType> > grid_refinement 
            = GridRefinement::GridRefinementFactory<dim,nstate,double,MeshType>::create_GridRefinement(gr_param,adjoint,physics_double);

        // starting the iterations
        dealii::ConvergenceTable convergence_table;
        dealii::Vector<float> estimated_error_per_cell(grid->n_active_cells());
        
        // for plotting the error convergence with gnuplot
        std::vector<double> solution_error;
        std::vector<double> functional_error;
        std::vector<double> dofs;

        for(unsigned int igrid = 0; igrid < refinement_steps; ++igrid){
            if(igrid > 0){
                grid_refinement->refine_grid();

                // option to quit here for testing mesh output
                if(gr_param.exit_after_refine)
                    continue;
            }

            // outputting the grid information
            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << refinement_steps
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            /* temp, zeroing solution */
            // {
            //     dg->solution *= 0.;
            // }

            // solving the system
            // option of whether to solve the problem or interpolate it from the manufactured solution
            auto solve_start = std::chrono::steady_clock::now();
            if(!grs_param.use_interpolation){
                ode_solver->steady_state();
            }else{
                dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
                solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
                dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
                dg->solution = solution_no_ghost;
            }
            auto solve_end = std::chrono::steady_clock::now();
            auto solve_diff = solve_end - solve_start;

            // TODO: computing necessary values
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->max_degree+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double linf_norm = 0.0;
            double l2_norm = 0.0;

            std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
            
            error_per_cell.resize(n_global_active_cells);

            for(auto cell = dg->dof_handler.begin_active(); cell < dg->dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                fe_values_extra.reinit(cell);
                cell->get_dof_indices(dofs_indices);

                double cell_l2error = 0.0;
                std::array<double,nstate> cell_linf;
                std::fill(cell_linf.begin(), cell_linf.end(), 0);

                for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for(unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof){
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    for(unsigned int istate = 0; istate < nstate; ++ istate){
                        const double uexact = physics_double->manufactured_solution_function->value(qpoint, istate);
                        cell_l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                        cell_linf[istate] = std::max(cell_linf[istate], abs(soln_at_q[istate]-uexact));
                    }
                }

                error_per_cell[cell->active_cell_index()] = l2_norm;

                l2_norm += cell_l2error;
                for(unsigned int istate = 0; istate < nstate; ++ istate){
                    linf_norm = std::max(linf_norm, cell_linf[istate]);
                }
            }
            const double l2_norm_mpi = std::sqrt(dealii::Utilities::MPI::sum(l2_norm, mpi_communicator));
            const double linf_norm_mpi = dealii::Utilities::MPI::max(linf_norm, mpi_communicator);

            // computing the functional value
            double functional_value = functional->evaluate_functional();

            // getting the functional error from the approximated fine grid functional value
            double func_error = abs(functional_value - functional_value_exact);

            // reinitializing the adjoint
            adjoint->reinit();

            // optional output of the adjoint results to vtk
            if(grs_param.output_adjoint_vtk){
                // evaluating the derivatives and the fine grid adjoint
                if(dg->get_max_fe_degree() + 1 <= dg->max_degree){ // don't output if at max order (as p-enrichment will segfault)
                    auto adj_start = std::chrono::steady_clock::now();
                    adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double,MeshType>::AdjointStateEnum::fine);
                    adjoint->fine_grid_adjoint();
                    estimated_error_per_cell.reinit(grid->n_active_cells());
                    estimated_error_per_cell = adjoint->dual_weighted_residual();
                    auto adj_end = std::chrono::steady_clock::now();
                    auto adj_diff = adj_end - adj_start;

                    // entries for timing
                    if(grs_param.output_adjoint_time){
                        convergence_table.add_value("adj_time", ((double)std::chrono::duration_cast<std::chrono::milliseconds>(adj_diff).count())/1000.);
                    }

                    adjoint->output_results_vtk(iref*10+igrid);
                }

                // and for the coarse grid
                adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double,MeshType>::AdjointStateEnum::coarse); // this one is necessary though
                adjoint->coarse_grid_adjoint();
                adjoint->output_results_vtk(iref*10+igrid);
            }

            // outputting the results from the grid refinement method
            if(grs_param.output_vtk)
                grid_refinement->output_results_vtk(iref);

            // convergence table
            if(grs_param.output_solution_error || grs_param.output_functional_error){
                convergence_table.add_value("cells", n_global_active_cells);
                convergence_table.add_value("DoFs", n_dofs);
            }

            // entries corresponding to the functional error
            if(grs_param.output_functional_error){
                convergence_table.add_value("value", functional_value);
                convergence_table.add_value("func_error", func_error);
            }

            // entries corresponding to the solution error
            if(grs_param.output_solution_error){
                convergence_table.add_value("l2_error", l2_norm_mpi);
                convergence_table.add_value("linf_error", linf_norm_mpi);
            }

            // entries for timing
            if(grs_param.output_solution_time){
                convergence_table.add_value("solve_time", ((double)std::chrono::duration_cast<std::chrono::milliseconds>(solve_diff).count())/1000.);
            }

            solution_error.push_back(l2_norm_mpi);
            functional_error.push_back(func_error);
            
            dofs.push_back(n_dofs);

            // temp
            PHiLiP::GridRefinement::MshOut<dim,double> msh_out(dg->dof_handler);
            std::string write_msh_name = "test-msh-" + dealii::Utilities::int_to_string(iref*10+igrid) + ".msh";
            std::ofstream out_msh(write_msh_name);
            msh_out.add_data_vector(error_per_cell, PHiLiP::GridRefinement::StorageType::element);
            msh_out.write_msh(out_msh);
        }

        if(grs_param.output_solution_error || grs_param.output_functional_error){
            pcout << " ********************************************" << std::endl
                << " Convergence rates for p = " << poly_degree << std::endl
                << " ********************************************" << std::endl;
        }

        // entries corresponding to the functional
        if(grs_param.output_functional_error){
            convergence_table.evaluate_convergence_rates("value", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("value", true);
            convergence_table.evaluate_convergence_rates("func_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("func_error", true);
        }

        // entries corresponding to the solution
        if(grs_param.output_solution_error){
            convergence_table.evaluate_convergence_rates("l2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("l2_error", true);
            convergence_table.evaluate_convergence_rates("linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("linf_error", true);
        }

        if(pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);
    
        // adding the data to gnuplot figure
        if(pcout.is_active() && grs_param.output_gnuplot_solution){
            gf_solution.add_xy_data(dofs, solution_error, "l2error."+dealii::Utilities::int_to_string(iref,1));
        
            // if refresh flag is set, output the plot at each iteration 
            if(grs_param.refresh_gnuplot)
                output_gnufig_solution(gf_solution);
        }

        if(pcout.is_active() && grs_param.output_gnuplot_functional){
            gf_functional.add_xy_data(dofs, functional_error, "ferror."+dealii::Utilities::int_to_string(iref,1));

            // if refresh flag is set, output the plot at each iteration
            if(grs_param.refresh_gnuplot)
                output_gnufig_functional(gf_functional);
        }

    }

    pcout << std::endl << std::endl << std::endl << std::endl
          << " ********************************************" << std::endl
          << " Convergence summary" << std::endl
          << " ********************************************" << std::endl;
    for(auto conv = convergence_table_vector.begin(); conv != convergence_table_vector.end(); ++conv){
        if(pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    if(pcout.is_active() && grs_param.output_gnuplot_solution)
        output_gnufig_solution(gf_solution);

    if(pcout.is_active() && grs_param.output_gnuplot_functional)
        output_gnufig_functional(gf_functional);

    return 0;
}

// gets the grid from the enum and reads file if neccesary
template <int dim, int nstate, typename MeshType>
void GridRefinementStudy<dim,nstate,MeshType>::get_grid(
    const std::shared_ptr<MeshType>&            grid,
    const Parameters::GridRefinementStudyParam& grs_param) const
{
    const unsigned int grid_size = grs_param.grid_size;

    const double left  = grs_param.grid_left;
    const double right = grs_param.grid_right;

    // generating the mesh
    using GridEnum = Parameters::ManufacturedConvergenceStudyParam::GridEnum;

    // considering different cases
    if(grs_param.grid_type == GridEnum::hypercube){

        dealii::Point<dim,double> p_left;
        dealii::Point<dim,double> p_right;
        std::vector<unsigned int> repetitions;
        for(unsigned int i = 0; i < dim; ++i){
            p_left[i] = left;
            p_right[i] = right;
            repetitions.push_back(grid_size);
        }

        // subdivided cube
        bool colorize = true;
        dealii::GridGenerator::subdivided_hyper_rectangle(*grid, repetitions, p_left, p_right, colorize);
        // dealii::GridGenerator::subdivided_hyper_cube(*grid, grid_size, left, right);
        for(auto cell = grid->begin_active(); cell != grid->end(); ++cell){
            cell->set_material_id(9002);
            for(unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                if(cell->face(face)->at_boundary()){
                    // temporarily disable
                    // cell->face(face)->set_boundary_id(1000);
                    // std::cout << cell->face(face)->boundary_id() << ", " << cell->face(face)->center() << std::endl;
                }
        }
    
    }else if(grs_param.grid_type == GridEnum::read_grid){

        // input grid file
        std::string read_mshname = grs_param.input_grid;
        std::cout << "Reading grid from: " << read_mshname << std::endl;

        // performing the read from file
        std::ifstream in_msh(read_mshname);
        dealii::GridIn<dim,dim> grid_in;
                    
        grid_in.attach_triangulation(*grid);
        grid_in.read_msh(in_msh);

    }
}

// performs the approximation of the functional value using a refined grid with interpolation
template <int dim, int nstate, typename MeshType>
double GridRefinementStudy<dim,nstate,MeshType>::approximate_exact_functional(
    const std::shared_ptr<Physics::PhysicsBase<dim,nstate,double>>& physics_double,
    const std::shared_ptr<Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<double>>>& /* physics_adtype */,
    const Parameters::AllParameters& param,
    const Parameters::GridRefinementStudyParam& grs_param) const
{
    // temporary meshtype object
    std::shared_ptr<MeshType> grid_fine = 
        MeshFactory<MeshType>::create_MeshType(this->mpi_communicator);

    // getting the grid from GridEnum type
    get_grid(grid_fine, grs_param);

    // performing refinement
    const int N_ref = 5;
    std::cout << "Starting refinement." << std::endl;
    grid_fine->refine_global(N_ref);
    std::cout << "Finished refinement." << std::endl;

    const unsigned int poly_degree      = grs_param.poly_degree;
    const unsigned int poly_degree_max  = grs_param.poly_degree_max;
    const unsigned int poly_degree_grid = grs_param.poly_degree_grid;

    // building the discontinuous galerkin solver
    std::cout << "Creating fine dg." << std::endl;
    std::shared_ptr< DGBase<dim, double, MeshType> > dg_fine = 
        DGFactory<dim,double,MeshType>::create_discontinuous_galerkin(
            &param, 
            poly_degree,
            poly_degree_max,
            poly_degree_grid,
            grid_fine);
    dg_fine->allocate_system();

    // interpolating the solution from the manufactured solution
    std::cout << "Performing solution transfer." << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg_fine->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg_fine->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    dg_fine->solution = solution_no_ghost;

    // creating a functional
    std::cout << "Generating the functional." << std::endl;
    std::shared_ptr< Functional<dim,nstate,double,MeshType> > functional_fine
        = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(grs_param.functional_param, dg_fine);

    // getting the "exact" value using it
    std::cout << "Computing and returning approximation" << std::endl;
    return functional_fine->evaluate_functional();
}

// function to perform the formatted output to gnuplot (of the solution error)
void output_gnufig_solution(PHiLiP::GridRefinement::GnuFig<double> &gf)
{
    // formatting for the figure and outputting .gp
    gf.set_name("ErrorPlot");
    gf.set_title("Error Convergence, |u|_2 vs. Dofs");
    gf.set_x_label("# Dofs");
    gf.set_y_label("L2 Error, |u|_2");
    gf.set_grid(false);
    gf.set_x_scale_log(true);
    gf.set_y_scale_log(true);
    gf.set_legend(true);
    gf.write_gnuplot();

    // performing plotting
    gf.exec_gnuplot();
}

// function to perform the formatted output to gnuplot (of the funcitonal error)
void output_gnufig_functional(
    PHiLiP::GridRefinement::GnuFig<double> &gf)
{
    // formatting for the figure and outputting .gp
    gf.set_name("FunctionalPlot");
    gf.set_title("Functional Convergence, |J(u)-J_h(u_h)| vs. Dofs");
    gf.set_x_label("# Dofs");
    gf.set_y_label("Functional error, |J(u)-J_h(u_h)|");
    gf.set_grid(false);
    gf.set_x_scale_log(true);
    gf.set_y_scale_log(true);
    gf.set_legend(true);
    gf.write_gnuplot();

    // performing plotting
    gf.exec_gnuplot();
}

// mesh factory specializations
template <>
std::shared_ptr<dealii::Triangulation<PHILIP_DIM>>
MeshFactory<dealii::Triangulation<PHILIP_DIM>>::create_MeshType(const MPI_Comm /* mpi_communicator */)
{
    return std::make_shared<dealii::Triangulation<PHILIP_DIM>>();
        // new dealii::Triangulation<PHILIP_DIM>(
        //     typename dealii::Triangulation<PHILIP_DIM>::MeshSmoothing(
        //         dealii::Triangulation<PHILIP_DIM>::smoothing_on_refinement |
        //         dealii::Triangulation<PHILIP_DIM>::smoothing_on_coarsening)));
}

template <>
std::shared_ptr<dealii::parallel::shared::Triangulation<PHILIP_DIM>>
MeshFactory<dealii::parallel::shared::Triangulation<PHILIP_DIM>>::create_MeshType(const MPI_Comm mpi_communicator)
{
    return std::make_shared<dealii::parallel::shared::Triangulation<PHILIP_DIM>>(
        mpi_communicator,
        typename dealii::Triangulation<PHILIP_DIM>::MeshSmoothing(
            dealii::Triangulation<PHILIP_DIM>::smoothing_on_refinement |
            dealii::Triangulation<PHILIP_DIM>::smoothing_on_coarsening));
}

#if PHILIP_DIM != 1
template <>
std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>>
MeshFactory<dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::create_MeshType(const MPI_Comm mpi_communicator)
{
    return std::make_shared<dealii::parallel::distributed::Triangulation<PHILIP_DIM>>(
        mpi_communicator,
        typename dealii::Triangulation<PHILIP_DIM>::MeshSmoothing(
            dealii::Triangulation<PHILIP_DIM>::smoothing_on_refinement |
            dealii::Triangulation<PHILIP_DIM>::smoothing_on_coarsening));
}
#endif

template class GridRefinementStudy <PHILIP_DIM,1,dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,2,dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,3,dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,4,dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,5,dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinementStudy <PHILIP_DIM,1,dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,2,dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,3,dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,4,dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,5,dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class GridRefinementStudy <PHILIP_DIM,1,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,2,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,3,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,4,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementStudy <PHILIP_DIM,5,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace Tests

} // namespace PHiLiP