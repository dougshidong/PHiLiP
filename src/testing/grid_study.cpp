#include <stdlib.h>     /* srand, rand */
#include <iostream>

//#include <deal.II/base/convergence_table.h> //<-- included in header file

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "tests.h"
#include "mesh/grids/curved_periodic_grid.hpp"

#include "grid_study.h"

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

#include "grid_refinement/grid_refinement.h"
#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/size_field.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
GridStudy<dim,nstate>::GridStudy(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
void GridStudy<dim,nstate>
::initialize_perturbed_solution(DGBase<dim,double> &dg, const Physics::PhysicsBase<dim,nstate,double> &physics) const
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::VectorTools::interpolate(mapping, dg.dof_handler, *physics.manufactured_solution_function, solution_no_ghost);
    //solution_no_ghost *= 1.0+1e-3;
    //solution_no_ghost = 0.0;
    //int i = 0;
    //for (auto sol = solution_no_ghost.begin(); sol != solution_no_ghost.end(); ++sol) {
    //    *sol = (++i) * 0.01;
    //}
    dg.solution = solution_no_ghost;

//Alex hard code initial condition
#if 0
            const unsigned int n_dofs = dg.dof_handler.n_dofs();
            for(unsigned int idof=0; idof<n_dofs; idof++){
                dg.solution[idof] = 0.0;
            }
#endif


}
template <int dim, int nstate>
double GridStudy<dim,nstate>
::integrate_solution_over_domain(DGBase<dim,double> &dg) const
{
    pcout << "Evaluating solution integral..." << std::endl;
    double solution_integral = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    //int overintegrate = 5;
    //dealii::QGauss<dim> quad_extra(dg.fe_system.tensor_degree()+overintegrate);
    //dealii::FEValues<dim,dim> fe_values_extra(dg.mapping, dg.fe_system, quad_extra, 
    //        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    //dealii::MappingQ<dim,dim> mappingq_temp(dg.max_degree+1);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra, 
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    const bool linear_output = true;
    int exponent;
    if (linear_output) exponent = 1;
    else exponent = 2;

    // Integrate solution error and output error
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {

        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Interpolate solution to quadrature points
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }
            // Integrate solution
            for (int s=0; s<nstate; s++) {
                solution_integral += pow(soln_at_q[s], exponent) * fe_values_extra.JxW(iquad);
            }
        }

    }
    const double solution_integral_mpi_sum = dealii::Utilities::MPI::sum(solution_integral, mpi_communicator);
    return solution_integral_mpi_sum;
}

template<int dim, int nstate>
std::string GridStudy<dim,nstate>::
get_convergence_tables_baseline_filename(const Parameters::AllParameters *const param) const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param->manufactured_convergence_study_param;
    
    // Future code development: create get_pde_string(), get_conv_num_flux_string(), get_manufactured_solution_string() in appropriate classes
    std::string error_filename_baseline = "convergence_table"; // initial base name
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = param->pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::advection)            {pde_string = "advection";}
    if (pde_type == PDE_enum::advection_vector)     {pde_string = "advection_vector";}
    if (pde_type == PDE_enum::diffusion)            {pde_string = "diffusion";}
    if (pde_type == PDE_enum::convection_diffusion) {pde_string = "convection_diffusion";}
    if (pde_type == PDE_enum::burgers_inviscid)     {pde_string = "burgers_inviscid";}
    if (pde_type == PDE_enum::burgers_rewienski)    {pde_string = "burgers_rewienski";}
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}

    using CNF_enum = Parameters::AllParameters::ConvectiveNumericalFlux;
    const CNF_enum CNF_type = param->conv_num_flux_type;
    std::string conv_num_flux_string;
    if (CNF_type == CNF_enum::lax_friedrichs) {conv_num_flux_string = "lax_friedrichs";}
    if (CNF_type == CNF_enum::split_form)     {conv_num_flux_string = "split_form";}
    if (CNF_type == CNF_enum::roe)            {conv_num_flux_string = "roe";}
    if (CNF_type == CNF_enum::l2roe)          {conv_num_flux_string = "l2roe";}

    using DNF_enum = Parameters::AllParameters::DissipativeNumericalFlux;
    const DNF_enum DNF_type = param->diss_num_flux_type;
    std::string diss_num_flux_string;
    if (DNF_type == DNF_enum::symm_internal_penalty) {diss_num_flux_string = "symm_internal_penalty";}
    if (DNF_type == DNF_enum::bassi_rebay_2)         {diss_num_flux_string = "bassi_rebay_2";}

    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
    const ManufacturedSolutionEnum MS_type = manu_grid_conv_param.manufactured_solution_param.manufactured_solution_type;
    std::string manufactured_solution_string;
    if (MS_type == ManufacturedSolutionEnum::sine_solution)           {manufactured_solution_string = "sine_solution";}
    if (MS_type == ManufacturedSolutionEnum::cosine_solution)         {manufactured_solution_string = "cosine_solution";}
    if (MS_type == ManufacturedSolutionEnum::additive_solution)       {manufactured_solution_string = "additive_solution";}
    if (MS_type == ManufacturedSolutionEnum::exp_solution)            {manufactured_solution_string = "exp_solution";}
    if (MS_type == ManufacturedSolutionEnum::poly_solution)           {manufactured_solution_string = "poly_solution";}
    if (MS_type == ManufacturedSolutionEnum::even_poly_solution)      {manufactured_solution_string = "even_poly_solution";}
    if (MS_type == ManufacturedSolutionEnum::atan_solution)           {manufactured_solution_string = "atan_solution";}
    if (MS_type == ManufacturedSolutionEnum::boundary_layer_solution) {manufactured_solution_string = "boundary_layer_solution";}
    if (MS_type == ManufacturedSolutionEnum::s_shock_solution)        {manufactured_solution_string = "s_shock_solution";}
    if (MS_type == ManufacturedSolutionEnum::quadratic_solution)      {manufactured_solution_string = "quadratic_solution";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_1)        {manufactured_solution_string = "navah_solution_1";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_2)        {manufactured_solution_string = "navah_solution_2";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_3)        {manufactured_solution_string = "navah_solution_3";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_4)        {manufactured_solution_string = "navah_solution_4";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_5)        {manufactured_solution_string = "navah_solution_5";}

    error_filename_baseline += std::string("_") + std::to_string(dim) + std::string("d");
    error_filename_baseline += std::string("_") + pde_string;
    error_filename_baseline += std::string("_") + conv_num_flux_string;
    error_filename_baseline += std::string("_") + diss_num_flux_string;
    error_filename_baseline += std::string("_") + manufactured_solution_string;
    return error_filename_baseline;
}

template<int dim, int nstate>
void GridStudy<dim,nstate>::
write_convergence_table_to_output_file(
    const std::string error_filename_baseline,
    const dealii::ConvergenceTable convergence_table,
    const unsigned int poly_degree) const
{
    std::string error_filename = error_filename_baseline;
    std::string error_fileType = std::string("txt");
    error_filename += std::string("_") + std::string("p") + std::to_string(poly_degree);
    std::ofstream error_table_file(error_filename + std::string(".") + error_fileType);
    convergence_table.write_text(error_table_file);
}


template<int dim, int nstate>
int GridStudy<dim,nstate>
::run_test () const
{
    int test_fail = 0;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&param);

    // Evaluate solution integral on really fine mesh
    double exact_solution_integral;
    pcout << "Evaluating EXACT solution integral..." << std::endl;
    // Limit the scope of grid_super_fine and dg_super_fine
#if PHILIP_DIM==1
        using Triangulation = dealii::Triangulation<dim>;
#else
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
    {
        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids_input);
        std::shared_ptr<Triangulation> grid_super_fine = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
            MPI_COMM_WORLD,
#endif
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        dealii::GridGenerator::subdivided_hyper_cube(*grid_super_fine, n_1d_cells[n_grids_input-1]);
        //dealii::Point<dim> p1, p2;
        //const double DX = 3;
        //for (int d=0;d<dim;++d) {
        //    p1[d] = 0.0-DX;
        //    p2[d] = 1.0+DX;
        //}
        //const std::vector<unsigned int> repetitions(dim,n_1d_cells[n_grids_input-1]);
        //dealii::GridGenerator::subdivided_hyper_rectangle<dim,dim>(*grid_super_fine, repetitions, p1, p2);

        //grid_super_fine->clear();
        //const std::vector<unsigned int> n_subdivisions(dim,n_1d_cells[n_grids_input-1]);
        //PHiLiP::Grids::curved_periodic_sine_grid<dim,Triangulation>(*grid_super_fine, n_subdivisions);
        for (auto cell = grid_super_fine->begin_active(); cell != grid_super_fine->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
            }
        }

        std::shared_ptr < DGBase<dim, double> > dg_super_fine = DGFactory<dim,double>::create_discontinuous_galerkin(&param, p_end, grid_super_fine);
        dg_super_fine->allocate_system ();

        initialize_perturbed_solution(*dg_super_fine, *physics_double);
        if (manu_grid_conv_param.output_solution) {
            dg_super_fine->output_results_vtk(9999);
        }
        exact_solution_integral = integrate_solution_over_domain(*dg_super_fine);
        pcout << "Exact solution integral is " << exact_solution_integral << std::endl;
    }

    int n_flow_convergence_error = 0;
    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    std::string error_filename_baseline;
    if (manu_grid_conv_param.output_convergence_tables) {
        error_filename_baseline = get_convergence_tables_baseline_filename(&param);
    }

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input + 1;

        std::vector<double> soln_error(n_grids);
        std::vector<double> output_error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        // Note that Triangulation must be declared before DG
        // DG will be destructed before Triangulation
        // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
        // Otherwise, a Subscriptor error will occur
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
            MPI_COMM_WORLD,
#endif
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        dealii::Vector<float>  estimated_error_per_cell;
        dealii::Vector<double> estimated_error_per_cell_double;

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
            grid->clear();
            dealii::GridGenerator::subdivided_hyper_cube(*grid, n_1d_cells[igrid]);
            //dealii::Point<dim> p1, p2;
            //const double DX = 3;
            //for (int d=0;d<dim;++d) {
            //    p1[d] = 0.0-DX;
            //    p2[d] = 1.0+DX;
            //}
            //const std::vector<unsigned int> repetitions(dim,n_1d_cells[igrid]);
            //dealii::GridGenerator::subdivided_hyper_rectangle<dim,dim>(*grid, repetitions, p1, p2);

            for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
                // Set a dummy boundary ID
                cell->set_material_id(9002);
                for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
                }
            }
            //dealii::GridTools::transform (&warp, *grid);
            // Warp grid if requested in input file
            if (manu_grid_conv_param.grid_type == GridEnum::sinehypercube) dealii::GridTools::transform (&warp, *grid);
            

            //grid->clear();
            //const std::vector<unsigned int> n_subdivisions(dim,n_1d_cells[igrid]);
            //PHiLiP::Grids::curved_periodic_sine_grid<dim,Triangulation>(*grid, n_subdivisions);
            //for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            //    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            //        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
            //    }
            //}

            // // Generate hypercube
            // if ( igrid==0 && (manu_grid_conv_param.grid_type == GridEnum::hypercube || manu_grid_conv_param.grid_type == GridEnum::sinehypercube ) ) {

            //     grid->clear();
            //     dealii::GridGenerator::subdivided_hyper_cube(*, n_1d_cells[igrid]);
            //     for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            //         // Set a dummy boundary ID
            //         cell->set_material_id(9002);
            //         for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            //             if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
            //         }
            //     }
            //     // Warp grid if requested in input file
            //     if (manu_grid_conv_param.grid_type == GridEnum::sinehypercube) dealii::GridTools::transform (&warp, *grid);
            // } else {
            //     dealii::GridRefinement::refine_and_coarsen_fixed_number(*grid,
            //                                     estimated_error_per_cell,
            //                                     0.3,
            //                                     0.03);
            //     grid->execute_coarsening_and_refinement();
            // }

            //for (int i=0; i<5;i++) {
            //    int icell = 0;
            //    for (auto cell = grid->begin_active(grid->n_levels()-1); cell!=grid->end(); ++cell) {
            //        if (!cell->is_locally_owned()) continue;
            //        icell++;
            //        if (icell < 2) {
            //            cell->set_refine_flag();
            //        }
            //        //else if (icell%2 == 0) {
            //        //    cell->set_refine_flag();
            //        //} else if (icell%3 == 0) {
            //        //    //cell->set_coarsen_flag();
            //        //}
            //    }
            //    grid->execute_coarsening_and_refinement();
            //}

            // Distort grid by random amount if requested
            const double random_factor = manu_grid_conv_param.random_distortion;
            const bool keep_boundary = true;
            if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, *grid, keep_boundary);

            // Read grid if requested
            if (manu_grid_conv_param.grid_type == GridEnum::read_grid) {
                //std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                std::string read_mshname = manu_grid_conv_param.input_grids+std::to_string(igrid)+".msh";
                pcout<<"Reading grid: " << read_mshname << std::endl;
                std::ifstream inmesh(read_mshname);
                dealii::GridIn<dim,dim> grid_in;
                grid->clear();
                grid_in.attach_triangulation(*grid);
                grid_in.read_msh(inmesh);
            }
            // Output grid if requested
            if (manu_grid_conv_param.output_meshes) {
                std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                std::ofstream outmesh(write_mshname);
                dealii::GridOutFlags::Msh msh_flags(true, true);
                dealii::GridOut grid_out;
                grid_out.set_flags(msh_flags);
                grid_out.write_msh(*grid, outmesh);
            }

            // Show mesh if in 2D
            //std::string gridname = "grid-"+std::to_string(igrid)+".eps";
            //if (dim == 2) print_mesh_info (*grid, gridname);

            using FadType = Sacado::Fad::DFad<double>;

            // Create DG object using the factory
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, grid);
            dg->allocate_system ();
            //dg->evaluate_inverse_mass_matrices();
            //
            // PhysicsBase required for exact solution and output error

            initialize_perturbed_solution(*(dg), *(physics_double));

            // Create ODE solver using the factory and providing the DG object
            std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim
                 << "\t Polynomial degree p: " << poly_degree
                 << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Solve the steady state problem
            //ode_solver->initialize_steady_polynomial_ramping (poly_degree);
            const int flow_convergence_error = ode_solver->steady_state();
            if (flow_convergence_error) n_flow_convergence_error += 1;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->max_degree+overintegrate);
            //dealii::MappingQ<dim,dim> mappingq(dg->max_degree+1);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            // l2error for each state
            std::array<double,nstate> cell_l2error_state;
            std::array<double,nstate> l2error_state;
            for (int istate=0; istate<nstate; ++istate) {
                l2error_state[istate] = 0.0;
            }

            double l2error = 0;

            // Integrate solution error and output error

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            estimated_error_per_cell.reinit(grid->n_active_cells());
            estimated_error_per_cell_double.reinit(grid->n_active_cells());
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                double cell_l2error = 0;

                for (int istate=0; istate<nstate; ++istate) {
                    cell_l2error_state[istate] = 0.0;
                }

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    for (int istate=0; istate<nstate; ++istate) {
                        const double uexact = physics_double->manufactured_solution_function->value(qpoint, istate);
                        //l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);

                        cell_l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);

                        cell_l2error_state[istate] += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad); // TESTING
                    }
                }
                estimated_error_per_cell[cell->active_cell_index()] = cell_l2error;
                estimated_error_per_cell_double[cell->active_cell_index()] = cell_l2error;
                l2error += cell_l2error;

                for (int istate=0; istate<nstate; ++istate) {
                    l2error_state[istate] += cell_l2error_state[istate];
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));

            std::array<double,nstate> l2error_mpi_sum_state;
            for (int istate=0; istate<nstate; ++istate) {
                l2error_mpi_sum_state[istate] = std::sqrt(dealii::Utilities::MPI::sum(l2error_state[istate], mpi_communicator));
            }

            double solution_integral = integrate_solution_over_domain(*dg);

            if (manu_grid_conv_param.output_solution) {
                /*
                dg->output_results_vtk(igrid);

                std::string write_posname = "error-"+std::to_string(igrid)+".pos";
                std::ofstream outpos(write_posname);
                GridRefinement::GmshOut<dim,double>::write_pos(grid,estimated_error_per_cell_double,outpos);

                std::shared_ptr< GridRefinement::GridRefinementBase<dim,nstate,double> >  gr 
                    = GridRefinement::GridRefinementFactory<dim,nstate,double>::create_GridRefinement(param.grid_refinement_study_param.grid_refinement_param_vector[0],dg,physics_double);

                gr->refine_grid();

                // dg->output_results_vtk(igrid);
                */
            
                // Use gr->output_results_vtk(), which includes L2error per cell, instead of dg->output_results_vtk() as done above
                std::shared_ptr< GridRefinement::GridRefinementBase<dim,nstate,double> >  gr 
                   = GridRefinement::GridRefinementFactory<dim,nstate,double>::create_GridRefinement(param.grid_refinement_study_param.grid_refinement_param_vector[0],dg,physics_double);
                gr->output_results_vtk(igrid);
            }
            

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
            output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("residual", dg->get_residual_l2norm ());
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("output_error", output_error[igrid]);

            // add l2error for each state to the convergence table
            if (manu_grid_conv_param.add_statewise_solution_error_to_convergence_tables) {
                std::array<std::string,nstate> soln_L2_error_state_str;
                for (int istate=0; istate<nstate; ++istate) {
                    soln_L2_error_state_str[istate] = std::string("soln_L2_error_state") + std::string("_") + std::to_string(istate);
                    convergence_table.add_value(soln_L2_error_state_str[istate], l2error_mpi_sum_state[istate]);
                }
            }

            pcout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                 << " Residual: " << ode_solver->residual_norm
                 << std::endl;

            pcout << " output_exact: " << exact_solution_integral
                 << " output_discrete: " << solution_integral
                 << " output_error: " << output_error[igrid]
                 << std::endl;

            if (igrid > 0) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                const double slope_output_err = log(output_error[igrid]/output_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << soln_error[igrid-1]
                     << "  solution_error2 " << soln_error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl
                     << "  solution_integral_error1 " << output_error[igrid-1]
                     << "  solution_integral_error2 " << output_error[igrid]
                     << "  slope " << slope_output_err
                     << std::endl;
            }

            // update the table with additional grid
            convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("dx", true);
            convergence_table.set_scientific("residual", true);
            convergence_table.set_scientific("soln_L2_error", true);
            convergence_table.set_scientific("output_error", true);
            if (manu_grid_conv_param.add_statewise_solution_error_to_convergence_tables) {
                std::string test_str;
                for (int istate=0; istate<nstate; ++istate) {
                    test_str = std::string("soln_L2_error_state") + std::string("_") + std::to_string(istate);
                    convergence_table.set_scientific(test_str,true);
                }
            }

            if (manu_grid_conv_param.output_convergence_tables) {
                write_convergence_table_to_output_file(
                    error_filename_baseline,
                    convergence_table,
                    poly_degree);
            }
        }
        pcout << " ********************************************"
             << std::endl
             << " Convergence rates for p = " << poly_degree
             << std::endl
             << " ********************************************"
             << std::endl;

        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        double last_slope = 0.0;
        if ( n_grids > 1 ) {
            last_slope = log(soln_error[n_grids-1]/soln_error[n_grids-2])
                             / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        }
        double before_last_slope = last_slope;
        if ( n_grids > 2 ) {
            before_last_slope = log(soln_error[n_grids-2]/soln_error[n_grids-3])
                                / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        }
        const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        if(poly_degree == 0) slope_deficit_tolerance *= 2; // Otherwise, grid sizes need to be much bigger for p=0

        if (slope_diff < slope_deficit_tolerance) {
            pcout << std::endl
                 << "Convergence order not achieved. Average last 2 slopes of "
                 << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }
    int n_fail_poly = fail_conv_poly.size();
    if (n_fail_poly > 0) {
        for (int ifail=0; ifail < n_fail_poly; ++ifail) {
            const double expected_slope = fail_conv_poly[ifail]+1;
            const double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
            pcout << std::endl
                 << "Convergence order not achieved for polynomial p = "
                 << fail_conv_poly[ifail]
                 << ". Slope of "
                 << fail_conv_slop[ifail] << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
        }
    }
    if (n_fail_poly) test_fail += 1;
    test_fail += n_flow_convergence_error;
    if (n_flow_convergence_error) {
        pcout << std::endl
              << "Flow did not converge some some cases. Please check the residuals achieved versus the residual tolerance."
              << std::endl;
    }
    return test_fail;
}

template <int dim, int nstate>
dealii::Point<dim> GridStudy<dim,nstate>
::warp (const dealii::Point<dim> &p)
{
    dealii::Point<dim> q = p;
    q[dim-1] *= 1.5;
    if (dim >= 2) q[0] += 1*std::sin(q[dim-1]);
    if (dim >= 3) q[1] += 1*std::cos(q[dim-1]);
    return q;
}

template <int dim, int nstate>
void GridStudy<dim,nstate>
::print_mesh_info(const dealii::Triangulation<dim> &triangulation, const std::string &filename) const
{
    pcout << "Mesh info:" << std::endl
         << " dimension: " << dim << std::endl
         << " no. of cells: " << triangulation.n_global_active_cells() << std::endl;
    std::map<dealii::types::boundary_id, unsigned int> boundary_count;
    for (auto cell : triangulation.active_cell_iterators()) {
        for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) boundary_count[cell->face(face)->boundary_id()]++;
        }
    }
    pcout << " boundary indicators: ";
    for (const std::pair<const dealii::types::boundary_id, unsigned int> &pair : boundary_count) {
        pcout      << pair.first << "(" << pair.second << " times) ";
    }
    pcout << std::endl;
    if (dim == 2) {
        std::ofstream out (filename);
        dealii::GridOut grid_out;
        grid_out.write_eps (triangulation, out);
        pcout << " written to " << filename << std::endl << std::endl;
    }
}

template class GridStudy <PHILIP_DIM,1>;
template class GridStudy <PHILIP_DIM,2>;
template class GridStudy <PHILIP_DIM,3>;
template class GridStudy <PHILIP_DIM,4>;
template class GridStudy <PHILIP_DIM,5>;
//template struct Instantiator<GridStudy,3,5>;



} // Tests namespace
} // PHiLiP namespace
