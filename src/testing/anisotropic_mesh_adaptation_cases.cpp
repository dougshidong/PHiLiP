#include <stdlib.h>
#include <iostream>
#include "physics/euler.h"
#include "anisotropic_mesh_adaptation_cases.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/anisotropic_mesh_adaptation.h"
#include "mesh/mesh_adaptation/fe_values_shape_hessian.h"
#include "mesh/mesh_adaptation/mesh_error_estimate.h"
#include "mesh/mesh_adaptation/mesh_optimizer.hpp"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/convergence_table.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
AnisotropicMeshAdaptationCases<dim, nstate> :: AnisotropicMeshAdaptationCases(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::write_solution_volume_nodes_to_file(std::shared_ptr<DGBase<dim,double>> dg) const
{
    dg->high_order_grid->volume_nodes.update_ghost_values();
    dg->solution.update_ghost_values();
    const int n_cells = dg->triangulation->n_global_active_cells();
    const int poly_degree = dg->get_min_fe_degree();
    const std::string filename_soln = 
                    "solution_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const std::string filename_volnodes = 
                    "volnodes_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const dealii::IndexSet &soln_range = dg->solution.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &vol_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    std::ofstream outfile_soln(filename_soln);
    std::ofstream outfile_volnodes(filename_volnodes);

    if( (!outfile_soln.is_open()) || (!outfile_volnodes.is_open()))
    {
        std::cout<<"Could not open file. Aborting.."<<std::endl;
        std::abort();
    }

    for(const auto &isol : soln_range)
    {
        outfile_soln<<std::setprecision(16)<<dg->solution(isol)<<"\n";
    }
    for(const auto &ivol : vol_range)
    {
        outfile_volnodes<<std::setprecision(16)<<dg->high_order_grid->volume_nodes(ivol)<<"\n";
    }
    outfile_soln.close();
    outfile_volnodes.close();
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::read_solution_volume_nodes_from_file(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const int n_cells = dg->triangulation->n_global_active_cells();
    const int poly_degree = dg->get_min_fe_degree();
    const std::string filename_soln = 
                    "solution_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const std::string filename_volnodes = 
                    "volnodes_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const dealii::IndexSet &soln_range = dg->solution.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &vol_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    std::ifstream infile_soln(filename_soln);
    std::ifstream infile_volnodes(filename_volnodes);

    if( (!infile_soln.is_open()) || (!infile_volnodes.is_open()))
    {
        std::cout<<"Could not open file. Aborting.."<<std::endl;
        std::abort();
    }

    for(const auto &isol : soln_range)
    {
        infile_soln>>dg->solution(isol);
    }
    for(const auto &ivol : vol_range)
    {
        infile_volnodes>>dg->high_order_grid->volume_nodes(ivol);
    }
    infile_soln.close();
    infile_volnodes.close();
    dg->high_order_grid->volume_nodes.update_ghost_values();
    dg->solution.update_ghost_values();
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::increase_grid_degree_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const unsigned int grid_degree_updated = 2;
    dg->high_order_grid->set_q_degree(grid_degree_updated, true);

    const unsigned int poly_degree_updated = dg->all_parameters->flow_solver_param.max_poly_degree_for_adaptation - 1;
    dg->set_p_degree_and_interpolate_solution(poly_degree_updated);
    project_surface_nodes_on_cylinder(dg);
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::project_surface_nodes_on_cylinder(std::shared_ptr<DGBase<dim,double>> dg) const
{
   const unsigned int n_surf_nodes = dg->high_order_grid->surface_nodes.size();
    const dealii::IndexSet &surface_range = dg->high_order_grid->surface_nodes.get_partitioner()->locally_owned_range();
    for(unsigned int i_surf = 0; i_surf < n_surf_nodes; ++i_surf)
    {
        if(!(surface_range.is_element(i_surf))) continue;
        const unsigned int vol_index = dg->high_order_grid->surface_to_volume_indices(i_surf);
        if(vol_index%dim==0)
        {
            const double x = dg->high_order_grid->volume_nodes(vol_index);
            const double y = dg->high_order_grid->volume_nodes(vol_index+1);
            const double norm_val  = sqrt(x*x + y*y); 
            const double radius = 1.0;
            if( abs(norm_val - radius) < 0.05)
            {
                dg->high_order_grid->volume_nodes(vol_index) = radius*x/norm_val;
                dg->high_order_grid->volume_nodes(vol_index+1) = radius*y/norm_val;
            }
        }
    }
    dg->high_order_grid->volume_nodes.update_ghost_values();
}
    
template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::refine_mesh_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const
{
    auto mesh_adaptation_param2 = dg->all_parameters->mesh_adaptation_param;
    mesh_adaptation_param2.use_goal_oriented_mesh_adaptation = false;
    mesh_adaptation_param2.refine_fraction = 1.0;
    mesh_adaptation_param2.h_coarsen_fraction = 0.0;
    std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
    std::make_unique<MeshAdaptation<dim,double>>(dg, &(mesh_adaptation_param2));
    meshadaptation->adapt_mesh();
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::test_numerical_flux(const std::shared_ptr<DGBase<dim,double>> dg) const
{
if constexpr (nstate==dim+2)
{
    std::vector<std::pair<double,double>> control_nodes_list;
    std::ifstream infile;
    std::string filepath;
    filepath = "q2_cylinder_controlnodes.txt";
    infile.open(filepath);
    if(!infile) {
        std::cout << "Could not open file in AnisotropicMeshAdaptationCases<dim,nstate>::test_numerical_flux."<< filepath << std::endl;
        std::abort();
    }

    std::string line;
    std::getline(infile, line); // skip the first line.
    while(std::getline(infile, line))
    {
        std::stringstream ss(line);

        std::string field_x;

        std::getline(ss, field_x, ',');

        std::stringstream ss_x(field_x);
        double xval = 0.0;
        ss_x >> xval;

        std::string field_y;

        std::getline(ss, field_y, ',');
        
        std::stringstream ss_y(field_y);

        double yval = 0;

        ss_y >> yval;

        control_nodes_list.push_back(std::make_pair(xval, yval));
    }
    std::cout<<"Verifying if the flux is upwinding at mesh face on the shock."<<std::endl;
    std::shared_ptr<Physics::Euler<dim,nstate,double>> euler_physics_double
        = std::make_shared<Physics::Euler<dim, nstate, double>>(
                dg->all_parameters->euler_param.ref_length,
                dg->all_parameters->euler_param.gamma_gas,
                dg->all_parameters->euler_param.mach_inf,
                dg->all_parameters->euler_param.angle_of_attack,
                dg->all_parameters->euler_param.side_slip_angle);
    std::unique_ptr < NumericalFlux::NumericalFluxConvective<dim, nstate, double > > conv_num_flux_double
     = NumericalFlux::NumericalFluxFactory<dim, nstate, double> ::create_convective_numerical_flux (dg->all_parameters->conv_num_flux_type, dg->all_parameters->pde_type, dg->all_parameters->model_type, euler_physics_double);
    
    dg->solution.update_ghost_values();
    const unsigned int poly_degree = dg->get_min_fe_degree();
    
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEFaceValues<dim,dim> fe_face_values_int(mapping, dg->fe_collection[poly_degree], 
                                dg->face_quadrature_collection[poly_degree],
                                dealii::update_normal_vectors | dealii::update_values | dealii::update_quadrature_points); 
    dealii::FEFaceValues<dim,dim> fe_face_values_ext(mapping, dg->fe_collection[poly_degree], 
                                dg->face_quadrature_collection[poly_degree],
                                dealii::update_normal_vectors | dealii::update_values | dealii::update_quadrature_points); 
    dealii::QGaussLobatto<dim-1> face_quad_GLL (2);
    dealii::FEFaceValues<dim,dim> fe_face_values_gll(mapping, dg->fe_collection[poly_degree], face_quad_GLL, dealii::update_quadrature_points); 
    const unsigned int n_face_quad_pts = fe_face_values_int.n_quadrature_points;
    const unsigned int n_dofs_cell = fe_face_values_int.dofs_per_cell;
    std::array<double,nstate> soln_int_at_q;
    std::array<double,nstate> soln_ext_at_q;
    std::vector<dealii::types::global_dof_index> dofs_indices_int(n_dofs_cell);
    std::vector<dealii::types::global_dof_index> dofs_indices_ext(n_dofs_cell);
    int n_quads_not_on_shock_upwinding = 0;
    int n_quads_on_shock_not_upwinding = 0;
    int n_shock_faces = 0;
    for(const auto &cell: dg->dof_handler.active_cell_iterators())
    {
        if(!cell->is_locally_owned()){continue;}
        cell->get_dof_indices(dofs_indices_int);
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface)
        {
            auto current_face = cell->face(iface);
            if(current_face->at_boundary()){continue;}
            const auto neighbor_cell = cell->neighbor_or_periodic_neighbor(iface);
            if(!dg->current_cell_should_do_the_work(cell, neighbor_cell)){continue;}
            fe_face_values_gll.reinit(cell,iface);
            const bool face_is_on_shock = is_face_between_control_nodes(
                    fe_face_values_gll.quadrature_point(0), 
                    fe_face_values_gll.quadrature_point(fe_face_values_gll.n_quadrature_points-1),
                    control_nodes_list);
            if(face_is_on_shock){n_shock_faces++;} else {continue;}
            bool is_upwinding = false;
            const unsigned int neighbor_iface = cell->neighbor_of_neighbor(iface);
            neighbor_cell->get_dof_indices(dofs_indices_ext);
            fe_face_values_int.reinit(cell,iface);
            fe_face_values_ext.reinit(neighbor_cell,neighbor_iface);

            for(unsigned int iquad = 0; iquad < n_face_quad_pts; ++iquad)
            {
                soln_int_at_q.fill(0.0); 
                soln_ext_at_q.fill(0.0);
                for(unsigned int idof = 0; idof < fe_face_values_int.dofs_per_cell; ++idof)
                {
                    const unsigned int istate = fe_face_values_int.get_fe().system_to_component_index(idof).first;
                    soln_int_at_q[istate] += dg->solution(dofs_indices_int[idof])*fe_face_values_int.shape_value_component(idof, iquad, istate);
                }
                for(unsigned int idof = 0; idof < fe_face_values_ext.dofs_per_cell; ++idof)
                {
                    const unsigned int istate = fe_face_values_ext.get_fe().system_to_component_index(idof).first;
                    soln_ext_at_q[istate] += dg->solution(dofs_indices_ext[idof])*fe_face_values_ext.shape_value_component(idof, iquad, istate);
                }
                const dealii::Tensor<1,dim,double> & normal_int = fe_face_values_int.normal_vector(iquad);

                std::array<double, nstate> numerical_flux_dot_n = conv_num_flux_double->evaluate_flux(soln_int_at_q, soln_ext_at_q, normal_int);

                const double mach_int = euler_physics_double->compute_mach_number(soln_int_at_q);
                const double mach_ext = euler_physics_double->compute_mach_number(soln_ext_at_q);
                std::array<double, nstate> expected_flux_dot_n;
                for(int istate=0; istate<nstate; ++istate)
                {
                    expected_flux_dot_n[istate] = 0;
                }
                if(mach_int>mach_ext)
                {
                    std::array<dealii::Tensor<1,dim,double>,nstate> flux_int = euler_physics_double->convective_flux(soln_int_at_q);
                    for(int d=0; d<dim; ++d)
                    {
                        for(int istate=0; istate<nstate; ++istate)
                        {
                            expected_flux_dot_n[istate] += flux_int[istate][d]*normal_int[d]; 
                        }
                    }                
                }
                else
                {
                    std::array<dealii::Tensor<1,dim,double>,nstate> flux_ext = euler_physics_double->convective_flux(soln_ext_at_q);
                    for(int d=0; d<dim; ++d)
                    {
                        for(int istate=0; istate<nstate; ++istate)
                        {
                            expected_flux_dot_n[istate] += flux_ext[istate][d]*normal_int[d]; 
                        }
                    }
                }

                std::array<double, nstate> error_flux;
                for(int istate=0; istate<nstate; ++istate)
                {
                    error_flux[istate] = abs(expected_flux_dot_n[istate] - numerical_flux_dot_n[istate]);
                }
                double error_flux_norm = 0;
                for(int istate=0; istate<nstate;++istate)
                {
                    error_flux_norm += pow(error_flux[istate],2);
                }
                error_flux_norm = sqrt(error_flux_norm);
                if(error_flux_norm < 1.0e-12)
                {
                    is_upwinding = true;
                }

                if(face_is_on_shock && (!is_upwinding))
                {
                    std::cout<<"======================================================================="<<std::endl;
                    std::cout<<"Location: "<<fe_face_values_int.quadrature_point(iquad)<<std::endl;
                    std::cout<<"On shock but not upwinding."<<std::endl;
                    std::cout<<"Difference in flux = "<<error_flux_norm<<std::endl;
                    ++n_quads_on_shock_not_upwinding;
                    std::cout<<"======================================================================="<<std::endl;
                }
                if(!face_is_on_shock && is_upwinding)
                {
                    std::cout<<"Location: "<<fe_face_values_int.quadrature_point(iquad)<<std::endl;
                    std::cout<<"Not on shock but still upwinding."<<std::endl;
                    ++n_quads_not_on_shock_upwinding;
                }
            } //iquad 
        } //iface
    } //cell
    pcout<<"Total number of shock faces = "<<dealii::Utilities::MPI::sum(n_shock_faces, this->mpi_communicator)<<std::endl;
    pcout<<"Total number of quads on shock but not upwinding = "<<dealii::Utilities::MPI::sum(n_quads_on_shock_not_upwinding, this->mpi_communicator)<<std::endl;
    pcout<<"Total number of quads not on shock but still upwinding = "<<dealii::Utilities::MPI::sum(n_quads_not_on_shock_upwinding, this->mpi_communicator)<<std::endl;
} // if constexpr
}

template<int dim, int nstate>
bool AnisotropicMeshAdaptationCases<dim,nstate>::is_face_between_control_nodes(
    const dealii::Point<dim> &point1, 
    const dealii::Point<dim> &point2, 
    const std::vector<std::pair<double,double>> &control_nodes_list) const
{
    bool point1_is_in_list = false;
    bool point2_is_in_list = false;

    for(unsigned int i=0; i<control_nodes_list.size(); ++i)
    {
        dealii::Point<dim> point_i;
        point_i[0] = control_nodes_list[i].first;
        point_i[1] = control_nodes_list[i].second;
        if(point_i.distance(point1) < 1.0e-4)
        {
            point1_is_in_list = true;
        }
        if(point_i.distance(point2) < 1.0e-4)
        {
            point2_is_in_list = true;
        }
    }
    return (point1_is_in_list && point2_is_in_list);
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::evaluate_regularization_matrix(
    dealii::TrilinosWrappers::SparseMatrix &regularization_matrix, 
    std::shared_ptr<DGBase<dim,double>> dg) const
{
    // Get volume of smallest element.
    const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[dg->high_order_grid->grid_degree];
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_vol(mapping, dg->high_order_grid->fe_metric_collection[dg->high_order_grid->grid_degree], volume_quadrature,
                    dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    const unsigned int n_quad_pts = fe_values_vol.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_values_vol.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_vol.dofs_per_cell);
    
    double min_cell_volume_local = 1.0e6;
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        double cell_vol = 0.0;
        fe_values_vol.reinit (cell);

        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }

        if(cell_vol < min_cell_volume_local)
        {
            min_cell_volume_local = cell_vol;
        }
    }

    const double min_cell_vol = dealii::Utilities::MPI::min(min_cell_volume_local, mpi_communicator);

    // Set sparsity pattern
    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dg->high_order_grid->dof_handler_grid,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    dealii::DynamicSparsityPattern dsp(dg->high_order_grid->dof_handler_grid.n_dofs(), dg->high_order_grid->dof_handler_grid.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dg->high_order_grid->dof_handler_grid, dsp, hanging_node_constraints);
    const dealii::IndexSet &locally_owned_dofs = dg->high_order_grid->locally_owned_dofs_grid;
    regularization_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, this->mpi_communicator);

    // Set elements.
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_vol.reinit (cell);
        cell->get_dof_indices(dofs_indices);
        cell_matrix = 0;
        
        double cell_vol = 0.0;
        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }
        const double omega_k = min_cell_vol/cell_vol;

        for(unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int icomp = fe_values_vol.get_fe().system_to_component_index(i).first;
            for(unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const unsigned int jcomp = fe_values_vol.get_fe().system_to_component_index(j).first;
                double val_ij = 0.0;

                if(icomp == jcomp)
                {
                    for(unsigned int q=0; q<n_quad_pts; ++q)
                    {
                        val_ij += omega_k*fe_values_vol.shape_grad(i,q)*fe_values_vol.shape_grad(j,q)*fe_values_vol.JxW(q);
                    }
                }
                cell_matrix(i,j) = val_ij;
            }
        }
        hanging_node_constraints.distribute_local_to_global(cell_matrix, dofs_indices, regularization_matrix); 
    } // cell loop ends
    regularization_matrix.compress(dealii::VectorOperation::add);
}

template <int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate> :: verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const
{
    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_jacobian_pushed_forward_grads | dealii::update_inverse_jacobians;
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, update_flags);
    
    dealii::MappingQGeneric<dim, dim> mapping2(dg.high_order_grid->get_current_fe_system().degree);
    dealii::hp::MappingCollection<dim> mapping_collection2(mapping2);
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume2 (mapping_collection2, dg.fe_collection, dg.volume_quadrature_collection, dealii::update_hessians);
    
    PHiLiP::FEValuesShapeHessian<dim> fe_values_shape_hessian;
    for(const auto &cell : dg.dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        const unsigned int i_mapp = 0;
        fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
        fe_values_collection_volume2.reinit(cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
        const dealii::FEValues<dim,dim> &fe_values_volume2 = fe_values_collection_volume2.get_present_fe_values();
        
        const unsigned int n_dofs_cell = fe_values_volume.dofs_per_cell;
        const unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;
        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            fe_values_shape_hessian.reinit(fe_values_volume, iquad);
            
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
                dealii::Tensor<2,dim,double> shape_hessian_dealii = fe_values_volume2.shape_hessian_component(idof, iquad, istate);
                
                dealii::Tensor<2,dim,double> shape_hessian_philip = fe_values_shape_hessian.shape_hessian_component(idof, iquad, istate, fe_values_volume.get_fe());

                dealii::Tensor<2,dim,double> shape_hessian_diff = shape_hessian_dealii;
                shape_hessian_diff -= shape_hessian_philip;

                if(shape_hessian_diff.norm() > 1.0e-8)
                {
                    std::cout<<"Dealii's FEValues shape_hessian = "<<shape_hessian_dealii<<std::endl;
                    std::cout<<"PHiLiP's FEValues shape_hessian = "<<shape_hessian_philip<<std::endl;
                    std::cout<<"Frobenius norm of diff = "<<shape_hessian_diff.norm()<<std::endl;
                    std::cout<<"Aborting.."<<std::endl<<std::flush;
                    std::abort();
                }
            } // idof
        } // iquad
    } // cell loop ends

    pcout<<"PHiLiP's physical shape hessian matches that computed by dealii."<<std::endl;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int countval) const
{
    const int outputval = 7000 + countval;
    dg->output_results_vtk(outputval);

    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    dg->assemble_residual();
    return abs_dwr_error;

    return 0;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const double functional_exact = (5.79555511474609375*2.0)*1.4*3.0;
    std::shared_ptr< Functional<dim, nstate, double> > functional
                                = FunctionalFactory<dim,nstate,double>::create_Functional(dg->all_parameters->functional_param, dg);
    const double functional_val = functional->evaluate_functional();
    const double error_val = abs(functional_val - functional_exact);
    return error_val;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_abs_dwr_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    dwr_error_val->total_dual_weighted_residual_error();
    return abs(dwr_error_val->net_functional_error);
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_enthalpy_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
if constexpr (nstate==dim+2)
{
    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                dg->all_parameters->euler_param.ref_length,
                dg->all_parameters->euler_param.gamma_gas,
                dg->all_parameters->euler_param.mach_inf,
                dg->all_parameters->euler_param.angle_of_attack,
                dg->all_parameters->euler_param.side_slip_angle);
    
    int overintegrate = 10;
    const unsigned int poly_degree = dg->get_min_fe_degree();
    dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
            dealii::update_values | dealii::update_JxW_values);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    const unsigned int n_dofs_cell = fe_values_extra.dofs_per_cell;
    std::array<double,nstate> soln_at_q;

    double l2error = 0;
    double l1error = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    // Integrate solution error and output error
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution(dofs_indices[idof])*fe_values_extra.shape_value_component(idof,iquad,istate); 
            }
            
            const double pressure = euler_physics_double.compute_pressure(soln_at_q);
            const double enthalpy_at_q = euler_physics_double.compute_specific_enthalpy(soln_at_q,pressure);
            l2error += pow((enthalpy_at_q - euler_physics_double.enthalpy_inf),2) * fe_values_extra.JxW(iquad);
            l1error += pow(euler_physics_double.compute_entropy_measure(soln_at_q) - euler_physics_double.entropy_inf,2) * fe_values_extra.JxW(iquad);
        }
    } // cell loop ends
    const double l2error_global = sqrt(dealii::Utilities::MPI::sum(l2error, MPI_COMM_WORLD));
    const double l1error_global = sqrt(dealii::Utilities::MPI::sum(l1error, MPI_COMM_WORLD));
    (void) l2error_global;
    (void) l1error_global;
    return l2error_global;
}
std::abort();
return 0.0;
}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
/*
    int output_val = 0;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);

    flow_solver->dg->freeze_artificial_dissipation=true;
    flow_solver->use_polynomial_ramping = false;
    increase_grid_degree_and_interpolate_solution(flow_solver->dg);
    read_solution_volume_nodes_from_file(flow_solver->dg);
    flow_solver->dg->assemble_residual();
    test_numerical_flux(flow_solver->dg);
    output_vtk_files(flow_solver->dg, output_val++);
    flow_solver->run();
    output_vtk_files(flow_solver->dg, output_val++);
    return 0;
*/

    int output_val = 0;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    const bool run_mesh_optimizer = param.optimization_param.max_design_cycles > 0;
    const bool run_fixedfraction_mesh_adaptation = param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0;
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);

    flow_solver->run();
    flow_solver->dg->freeze_artificial_dissipation=true;
    flow_solver->use_polynomial_ramping = false;
    output_vtk_files(flow_solver->dg, output_val++);
    //return 0;

    std::vector<double> functional_error_vector;
    std::vector<double> enthalpy_error_vector;
    std::vector<unsigned int> n_cycle_vector;
    std::vector<unsigned int> n_dofs_vector;

    const double functional_error_initial = evaluate_functional_error(flow_solver->dg);
    //pcout<<"Functional error initial = "<<std::setprecision(16)<<functional_error_initial<<std::endl; // can be deleted later.
    const double enthalpy_error_initial = evaluate_enthalpy_error(flow_solver->dg);
    functional_error_vector.push_back(functional_error_initial);
    enthalpy_error_vector.push_back(enthalpy_error_initial);
    n_dofs_vector.push_back(flow_solver->dg->n_dofs());
    unsigned int current_cycle = 0;
    n_cycle_vector.push_back(current_cycle++);
    dealii::ConvergenceTable convergence_table_functional;
    dealii::ConvergenceTable convergence_table_enthalpy;
    if(run_mesh_optimizer)
    {
        const bool use_oneD_parameteriation = true;
        flow_solver->dg->freeze_artificial_dissipation=true;

        // q1 initial run
        {
            flow_solver->dg->set_p_degree_and_interpolate_solution(1);
            /* Uncomment later
            dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q1;
            evaluate_regularization_matrix(regularization_matrix_poisson_q1, flow_solver->dg);
            Parameters::AllParameters param_q1 = param;
            param_q1.optimization_param.max_design_cycles = 4;
            param_q1.optimization_param.regularization_parameter_sim = 1.0;
            
            std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q1 = 
                            std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg, &param_q1, true);
            const bool output_refined_nodes = false;
            mesh_optimizer_q1->run_full_space_optimizer(regularization_matrix_poisson_q1, use_oneD_parameteriation, output_refined_nodes, output_val-1);
            */
            /*
            flow_solver->run();
            if(flow_solver->dg->get_residual_l2norm() > 1.0e-10)
            {
                std::cout<<"Residual from q1 optimization has not converged. Aborting..."<<std::endl;
                std::abort();
            }
            */
        }

        // q2 initial run
        {
            increase_grid_degree_and_interpolate_solution(flow_solver->dg);
    /*
            Parameters::AllParameters param_q2 = param;
            param_q2.optimization_param.max_design_cycles = 20;
            param_q2.optimization_param.regularization_parameter_sim = 1.0;
            dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q2;
            evaluate_regularization_matrix(regularization_matrix_poisson_q2, flow_solver->dg);
            flow_solver->dg->freeze_artificial_dissipation=true;
            std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q2 = std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg,&param_q2, true);
            const bool output_refined_nodes = false;
            mesh_optimizer_q2->run_full_space_optimizer(regularization_matrix_poisson_q2, use_oneD_parameteriation, output_refined_nodes);
            output_vtk_files(flow_solver->dg, output_val++);
        */
        }

        const unsigned int n_meshes = 4;
    /* Uncomment later
        for(unsigned int imesh = 0; imesh < n_meshes; ++imesh)
        {
            if(imesh>0)
            {
                refine_mesh_and_interpolate_solution(flow_solver->dg); 
            }
            dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q2;
            evaluate_regularization_matrix(regularization_matrix_poisson_q2, flow_solver->dg);
            flow_solver->dg->freeze_artificial_dissipation=true;
            flow_solver->dg->set_upwinding_flux(true);
            std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q2 = std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg,&param, true);
            const bool output_refined_nodes = true;
            mesh_optimizer_q2->run_full_space_optimizer(regularization_matrix_poisson_q2, use_oneD_parameteriation, output_refined_nodes, output_val);
            output_vtk_files(flow_solver->dg, output_val++);
            write_solution_volume_nodes_to_file(flow_solver->dg);

            const double functional_error = evaluate_functional_error(flow_solver->dg);
            const double enthalpy_error = evaluate_enthalpy_error(flow_solver->dg);
            functional_error_vector.push_back(functional_error);
            enthalpy_error_vector.push_back(enthalpy_error);
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);

            convergence_table_functional.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_functional.add_value("functional_error",functional_error);
            convergence_table_enthalpy.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_enthalpy.add_value("enthalpy_error",enthalpy_error);
        } //imesh loop
    */
        for(unsigned int imesh = 0; imesh < n_meshes; ++imesh)
        {
            if(imesh>0)
            {
                refine_mesh_and_interpolate_solution(flow_solver->dg); 
            }
            if(imesh==2)
            {
                (void) use_oneD_parameteriation;
                pcout<<"Reading mesh and solution from file."<<std::endl;
                read_solution_volume_nodes_from_file(flow_solver->dg);
                flow_solver->dg->set_upwinding_flux(true);
                pcout<<"Ncells = "<<flow_solver->dg->triangulation->n_global_active_cells()<<std::endl;
                pcout<<"Residual norm before = "<<flow_solver->dg->get_residual_l2norm()<<std::endl;
                pcout<<"Assembling residual."<<std::endl;
                flow_solver->dg->assemble_residual();
                pcout<<"Residual norm after = "<<flow_solver->dg->right_hand_side.l2_norm()<<std::endl;
                pcout<<"Outputting file."<<std::endl;
                output_vtk_files(flow_solver->dg, output_val++);
            }
            if(imesh==3)
            {
                pcout<<"Ncells = "<<flow_solver->dg->triangulation->n_global_active_cells()<<std::endl;
                dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q2;
                evaluate_regularization_matrix(regularization_matrix_poisson_q2, flow_solver->dg);
                flow_solver->dg->freeze_artificial_dissipation=true;
                flow_solver->dg->set_upwinding_flux(true);
                std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q2 = std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg,&param, true);
                const bool output_refined_nodes = true;
                mesh_optimizer_q2->run_full_space_optimizer(regularization_matrix_poisson_q2, use_oneD_parameteriation, output_refined_nodes, output_val);
                output_vtk_files(flow_solver->dg, output_val++);
                
            }
        }
    }

    if(run_fixedfraction_mesh_adaptation)
    {
        const unsigned int n_adaptation_cycles = param.mesh_adaptation_param.total_mesh_adaptation_cycles;

        std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
        std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(param.mesh_adaptation_param));

        for(unsigned int icycle = 0; icycle < n_adaptation_cycles; ++icycle)
        {
            meshadaptation->adapt_mesh();
            flow_solver->run();

            const double functional_error = evaluate_functional_error(flow_solver->dg);
            const double enthalpy_error = evaluate_enthalpy_error(flow_solver->dg);
            functional_error_vector.push_back(functional_error);
            enthalpy_error_vector.push_back(enthalpy_error);
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);

            convergence_table_functional.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_functional.add_value("functional_error",functional_error);
            convergence_table_enthalpy.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_enthalpy.add_value("enthalpy_error",enthalpy_error);
        }
    }

    output_vtk_files(flow_solver->dg, output_val++);

    // output error vals
    pcout<<"\n cycles = [";
    for(long unsigned int i=0; i<n_cycle_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<n_cycle_vector[i];
    }
    pcout<<"];"<<std::endl;

    pcout<<"\n n_dofs = [";
    for(long unsigned int i=0; i<n_dofs_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<n_dofs_vector[i];
    }
    pcout<<"];"<<std::endl;

    std::string functional_type = "functional_error";
    pcout<<"\n "<<functional_type<<" = ["<<std::setprecision(16);
    for(long unsigned int i=0; i<functional_error_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<functional_error_vector[i];
    }
    pcout<<"];"<<std::endl;
    
    std::string errortype = "enthalpy_error";
    pcout<<"\n "<<errortype<<" = ["<<std::setprecision(16);
    for(long unsigned int i=0; i<enthalpy_error_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<enthalpy_error_vector[i];
    }
    pcout<<"];"<<std::endl;

    convergence_table_functional.evaluate_convergence_rates("functional_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table_functional.set_scientific("functional_error", true);
    convergence_table_enthalpy.evaluate_convergence_rates("enthalpy_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table_enthalpy.set_scientific("enthalpy_error", true);

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary for functional error" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table_functional.write_text(pcout.get_stream());}
    
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary for enthalpy error" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table_enthalpy.write_text(pcout.get_stream());}

return 0;

}

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
