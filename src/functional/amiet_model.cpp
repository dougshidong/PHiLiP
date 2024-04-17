#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "amiet_model.hpp"

namespace PHiLiP {

//================================================================
// Amiet's model
//================================================================
template <int dim,int nstate,typename real,typename MeshType>
AmietModelFunctional<dim,nstate,real,MeshType>
::AmietModelFunctional(
    std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
    const ExtractionFunctional<dim,nstate,real,MeshType> & boundary_layer_extraction_input,
    const dealii::Point<3,real> & observer_coord_ref_input)
    : Functional<dim,nstate,real,MeshType>(dg_input)
    , acoustic_contribution_type(this->dg->all_parameters->amiet_param.acoustic_contribution_type)
    , wall_pressure_spectral_model_type(this->dg->all_parameters->amiet_param.wall_pressure_spectral_model_type)
    , boundary_layer_extraction(boundary_layer_extraction_input)
    , omega_min(this->dg->all_parameters->amiet_param.omega_min)
    , omega_max(this->dg->all_parameters->amiet_param.omega_max)
    , d_omega(this->dg->all_parameters->amiet_param.omega_interval)
    , numb_of_omega((omega_max-omega_min)/d_omega)
    , observer_coord_ref(observer_coord_ref_input)
    , R_specific(this->dg->all_parameters->amiet_param.R_specific)
    , ref_density(this->dg->all_parameters->amiet_param.ref_density)
    , ref_length(this->dg->all_parameters->euler_param.ref_length)
    , ref_temperature(this->dg->all_parameters->navier_stokes_param.temperature_inf)
    , mach_inf(this->dg->all_parameters->euler_param.mach_inf)
    , sound_inf(sqrt(this->dg->all_parameters->euler_param.gamma_gas*R_specific*ref_temperature))
    , ref_speed(mach_inf*sound_inf)
    , ref_kinematic_viscosity(ref_speed*ref_length/this->dg->all_parameters->navier_stokes_param.reynolds_number_inf)
    , chord_length(this->dg->all_parameters->amiet_param.chord_length*ref_length)
    , span_length(this->dg->all_parameters->amiet_param.span_length*ref_length)
    , alpha(this->dg->all_parameters->amiet_param.alpha)
    , b(chord_length/2.0)
    , beta_sqr(1.0-mach_inf*mach_inf)
    , S0(sqrt(observer_coord_ref[0]*observer_coord_ref[0]+beta_sqr*(observer_coord_ref[1]*observer_coord_ref[1]+observer_coord_ref[2]*observer_coord_ref[2])))
{
    std::complex<real> imag (0.0,1.0);
    imag_unit = imag;
 
    //std::cout << "ref_density is "                   << ref_density << std::endl;
    //std::cout << "ref_temperature is "               << ref_temperature << std::endl;
    //std::cout << "ref_U is "                         << ref_U << std::endl;
    //std::cout << "ref_viscosity is "                 << ref_viscosity << std::endl;

    //std::cout << "U_inf is "                         << U_inf << std::endl;
    //std::cout << "sound_inf is "                     << sound_inf << std::endl;
    //std::cout << "density_inf is "                   << density_inf << std::endl;
    //std::cout << "U_c is "                           << U_c << std::endl;
    //std::cout << "U_edge is "                        << U_edge << std::endl;
    //std::cout << "friction_velocity is "             << friction_velocity << std::endl;
    //std::cout << "boundary_layer_thickness is "      << boundary_layer_thickness << std::endl;
    //std::cout << "displacement_thickness is "        << displacement_thickness << std::endl;
    //std::cout << "momentum_thickness is "            << momentum_thickness << std::endl;
    //std::cout << "wall_shear_stress is "             << wall_shear_stress << std::endl;
    //std::cout << "maximum_shear_stress is "          << maximum_shear_stress << std::endl;
    //std::cout << "kinematic_viscosity is "           << kinematic_viscosity << std::endl;
    //std::cout << "pressure_gradient_tangential is "  << pressure_gradient_tangential << std::endl;
    //std::cout << "clauser_equilibrium_parameter is " << clauser_equilibrium_parameter << std::endl;
    //std::cout << "cole_wake_parameter is "           << cole_wake_parameter << std::endl;
    //std::cout << "zagarola_smits_parameter is "      << zagarola_smits_parameter << std::endl;

    //Phi_pp.resize(numb_of_omega);
    //S_pp.resize(numb_of_omega);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_functional(const bool compute_dIdW, 
                      const bool compute_dIdX, 
                      const bool compute_d2I)
{
    using FadType = Sacado::Fad::DFad<real>;
    using FadFadType = Sacado::Fad::DFad<FadType>;

    bool actually_compute_value = true;
    bool actually_compute_dIdW  = compute_dIdW;
    // Note: for Amiet's Model compute_dIdX should be false
    bool actually_compute_dIdX  = compute_dIdX;
    // Note: for reduced space method compute_d2I should be false
    bool actually_compute_d2I   = compute_d2I;

    this->pcout << "Evaluating functional... ";
    this->need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    this->pcout << std::endl;

    if (!actually_compute_value && !actually_compute_dIdW && !actually_compute_dIdX && !actually_compute_d2I) {
        return this->current_functional_value;
    }

    // To obtain the local derivatives
    // Todo: remove metric elements if not needed
    //const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid->fe_system;
    //const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    //std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);

    const dealii::hp::FECollection<dim> fe_collection = this->dg->dof_handler.get_fe_collection();

    std::vector<FadFadType> soln_coeff(max_dofs_per_cell); 

    const auto mapping = (*(this->dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    this->allocate_derivatives(actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);

    // local derivative of interpolated solutions wrt flow solutions
    std::vector<real> local_dW_int_i_dW;
    // local derivative of interpolated solutions gradient wrt flow solutions
    std::vector<std::vector<real>> local_dW_grad_int_i_dW(dim);

    // Todo: define this function
    //allocate_dW_int_dW();

    //dealii::Point<dim,real> extraction_point;
    //if constexpr(dim==2){
    //    extraction_point[0] = this->dg->all_parameters.boundary_layer_extraction_param.extraction_point_x;
    //    extraction_point[1] = this->dg->all_parameters.boundary_layer_extraction_param.extraction_point_y;
    //} else if constexpr(dim==3){
    //    extraction_point[0] = this->dg->all_parameters.boundary_layer_extraction_param.extraction_point_x;
    //    extraction_point[1] = this->dg->all_parameters.boundary_layer_extraction_param.extraction_point_y;
    //    extraction_point[2] = this->dg->all_parameters.boundary_layer_extraction_param.extraction_point_z;
    //}
    //// Note: number_of_sampling contains only the sampling quadrature points
    //int number_of_sampling = this->dg->all_parameters.boundary_layer_extraction_param.number_of_sampling;
    //// Note: number_of_total_sampling contains the start and end points as well as all sampling quadrature points
    //int number_of_total_sampling = number_of_sampling+2;
    //ExtractionFunctional<dim,nstate,real,MeshType> boundary_layer_extraction(this->dg, extraction_point, number_of_sampling);

    // coord_of_total_sampling: vector contains coordinates of start and end points as well as all sampling quadrature points
    int number_of_total_sampling = boundary_layer_extraction.number_of_total_sampling;
    std::vector<dealii::Point<dim,real>> coord_of_total_sampling;
    coord_of_total_sampling.resize(number_of_total_sampling);
    coord_of_total_sampling = boundary_layer_extraction.evaluate_straight_line_total_sampling_point_coord();

    std::vector<std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,typename dealii::Point<dim,real>>> cell_index_and_ref_points_of_total_sampling(number_of_total_sampling);

    cell_index_and_ref_points_of_total_sampling = boundary_layer_extraction.find_active_cell_around_points(mapping_collection,this->dg->dof_handler,coord_of_total_sampling);

    // Todo: MPI version of W_int and W_grad_int
    std::vector<std::array<FadType,nstate>> soln_of_total_sampling(number_of_total_sampling);
    std::vector<std::array<dealii::Tensor<1,dim,FadType>,nstate>> soln_grad_of_total_sampling(number_of_total_sampling);

    int n_total_int_indep = number_of_total_sampling*nstate;

    std::vector<std::vector<real>> dW_int_dW(n_total_int_indep,std::vector<real>(this->dg->dof_handler.n_dofs()));
    std::vector<std::vector<real>> dW_grad_int_dW(3*n_total_int_indep,std::vector<real>(this->dg->dof_handler.n_dofs()));

    this->dg->solution.update_ghost_values();
    //auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    auto soln_cell = this->dg->dof_handler.begin_active();
    for( ; soln_cell != this->dg->dof_handler.end(); ++soln_cell/*, ++metric_cell*/) {
        if(!soln_cell->is_locally_owned()) continue;

        unsigned int sampling_index;
        for(int i=0;i<number_of_total_sampling;++i){
            if(cell_index_and_ref_points_of_total_sampling[i].first == soln_cell){
                sampling_index = i;

                const unsigned int i_fele = soln_cell->active_fe_index();

                // Get solution coefficients
                const dealii::FESystem<dim,dim> &fe_solution = this->dg->fe_collection[i_fele];
                const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
                cell_soln_dofs_indices.resize(n_soln_dofs_cell);
                soln_cell->get_dof_indices(cell_soln_dofs_indices);
                soln_coeff.resize(n_soln_dofs_cell);

                // Get metric coefficients
                //metric_cell->get_dof_indices (cell_metric_dofs_indices);
                //std::vector< FadFadType > coords_coeff(n_metric_dofs_cell);
                //for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                //    coords_coeff[idof] = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
                //}

                // Setup automatic differentiation
                unsigned int n_total_indep = 0;
                if (actually_compute_dIdW /*|| actually_compute_d2I*/) n_total_indep += n_soln_dofs_cell;
                //if (actually_compute_dIdX || actually_compute_d2I) n_total_indep += n_metric_dofs_cell;
                unsigned int i_derivative = 0;
                for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
                    const real val = this->dg->solution[cell_soln_dofs_indices[idof]];
                    soln_coeff[idof] = val;
                    if (actually_compute_dIdW /*|| actually_compute_d2I*/)
                        soln_coeff[idof].diff(i_derivative++, n_total_indep);
                }
                //for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                //    const real val = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
                //    coords_coeff[idof] = val;
                //    if (actually_compute_dIdX || actually_compute_d2I) coords_coeff[idof].diff(i_derivative++, n_total_indep);
                //}
                AssertDimension(i_derivative, n_total_indep);
                //if (actually_compute_d2I) {
                    //unsigned int i_derivative = 0;
                    //for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
                    //    const real val = dg->solution[cell_soln_dofs_indices[idof]];
                    //    soln_coeff[idof].val() = val;
                    //    soln_coeff[idof].val().diff(i_derivative++, n_total_indep);
                    //}
                    //for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                    //    const real val = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
                    //    coords_coeff[idof].val() = val;
                    //    coords_coeff[idof].val().diff(i_derivative++, n_total_indep);
                    //}
                //}
                //AssertDimension(i_derivative, n_total_indep);

                std::array<FadFadType,nstate> soln_of_sampling;
                std::array<dealii::Tensor<1,dim,FadFadType>,nstate> soln_grad_of_sampling;

                soln_of_sampling = boundary_layer_extraction.point_value(coord_of_total_sampling[sampling_index],
                                                                         mapping_collection,
                                                                         fe_collection,
                                                                         cell_index_and_ref_points_of_total_sampling[sampling_index],
                                                                         soln_coeff,
                                                                         cell_soln_dofs_indices);

                soln_grad_of_sampling = boundary_layer_extraction.point_gradient(coord_of_total_sampling[sampling_index],
                                                                                 mapping_collection,
                                                                                 fe_collection,
                                                                                 cell_index_and_ref_points_of_total_sampling[sampling_index],
                                                                                 soln_coeff,
                                                                                 cell_soln_dofs_indices);

                for(unsigned int s=0;s<nstate;++s){
                    soln_of_total_sampling[sampling_index][s] = soln_of_sampling[s].val();
                }

                for(unsigned int s=0;s<nstate;++s){
                    for(unsigned int d=0;d<dim;++d){
                        soln_grad_of_total_sampling[sampling_index][s][d] = soln_grad_of_sampling[s][d].val();
                    }
                }

                // getting the values and adding them to the derivaitve vector
                if (actually_compute_dIdW) {
                    local_dW_int_i_dW.resize(n_soln_dofs_cell);
                    for(unsigned int s=0;s<nstate;++s){
                        i_derivative = 0;
                        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
                            local_dW_int_i_dW[idof] = soln_of_sampling[s].dx(i_derivative++).val();
                        }
                        AssertDimension(i_derivative, n_total_indep);
                        unsigned int global_int_dof_index = sampling_index*nstate+s;
                        //dW_int_dW->add(global_int_dof_index, cell_soln_dofs_indices, local_dW_int_i_dW);

                        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
                            dW_int_dW[global_int_dof_index][cell_soln_dofs_indices[idof]] = local_dW_int_i_dW[idof];
                        }
                    }

                    for(unsigned int d=0;d<dim;++d){
                        local_dW_grad_int_i_dW[d].resize(n_soln_dofs_cell);
                    }
                    for(unsigned int s=0;s<nstate;++s){
                        for(unsigned int d=0;d<dim;++d){
                            i_derivative = 0;
                            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
                                local_dW_grad_int_i_dW[d][idof] = soln_grad_of_sampling[s][d].dx(i_derivative++).val();
                            }
                            AssertDimension(i_derivative, n_total_indep);
                        }
                        unsigned int global_int_dof_index_x = 0*n_total_int_indep+sampling_index*nstate+s;
                        unsigned int global_int_dof_index_y = 1*n_total_int_indep+sampling_index*nstate+s;
                        unsigned int global_int_dof_index_z = 2*n_total_int_indep+sampling_index*nstate+s;
                        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
                            dW_grad_int_dW[global_int_dof_index_x][cell_soln_dofs_indices[idof]] = local_dW_grad_int_i_dW[0][idof];
                            dW_grad_int_dW[global_int_dof_index_y][cell_soln_dofs_indices[idof]] = local_dW_grad_int_i_dW[1][idof];
                            dW_grad_int_dW[global_int_dof_index_z][cell_soln_dofs_indices[idof]] = local_dW_grad_int_i_dW[2][idof];
                        }
                    }
                
                }
                if (actually_compute_dIdX) {
                    std::cout << "ERROR: No dependency of coordinate exists for Amiet's model..." << std::endl;
                }
                //if (actually_compute_dIdW || actually_compute_dIdX) AssertDimension(i_derivative, n_total_indep);
                if (actually_compute_d2I) {
                    std::cout << "ERROR: Full space method are not supported for Amiet's model..." << std::endl;
                }
                //AssertDimension(i_derivative, n_total_indep);
            }
        }
    }
    
    // Todo: maynot need compress operation due to no ghost cells are involved
    //if (actually_compute_dIdW) dW_int_dW->compress(dealii::VectorOperation::add);

    unsigned int i_int_derivative = 0;
    for(int int_i=0;int_i<number_of_total_sampling;++int_i){
        for(int s=0;s<nstate;++s){
            soln_of_total_sampling[int_i][s].diff(i_int_derivative++,n_total_int_indep+3*n_total_int_indep);
        }
    }
    AssertDimension(i_int_derivative, n_total_int_indep);
    for(int int_i=0;int_i<number_of_total_sampling;++int_i){
        for(int s=0;s<nstate;++s){
            for(int d=0;d<dim;++d){
                soln_grad_of_total_sampling[int_i][s][d].diff(i_int_derivative++,n_total_int_indep+3*n_total_int_indep);
            }
        }
    }
    AssertDimension(i_int_derivative, n_total_int_indep+3*n_total_int_indep);

    std::pair<real,real> values_free_stream = boundary_layer_extraction.evaluate_converged_free_stream_values(soln_of_total_sampling);

    real speed_free_stream = values_free_stream.first*ref_speed;
    real density_free_stream = values_free_stream.second*ref_density;
    real U_c = speed_free_stream/alpha;

    real boundary_layer_thickness = boundary_layer_extraction.evaluate_boundary_layer_thickness(coord_of_total_sampling,soln_of_total_sampling)*ref_length;
    real edge_velocity            = boundary_layer_extraction.evaluate_edge_velocity(soln_of_total_sampling)*ref_speed;
    real maximum_shear_stress     = boundary_layer_extraction.evaluate_maximum_shear_stress(soln_of_total_sampling,soln_grad_of_total_sampling)*ref_density*ref_speed*ref_speed;

    FadType displacement_thickness_fad       = boundary_layer_extraction.evaluate_displacement_thickness(soln_of_total_sampling)*ref_length;
    FadType momentum_thickness_fad           = boundary_layer_extraction.evaluate_momentum_thickness(soln_of_total_sampling)*ref_length;
    FadType friction_velocity_fad            = boundary_layer_extraction.evaluate_friction_velocity(soln_of_total_sampling,soln_grad_of_total_sampling)*ref_speed;
    FadType pressure_gradient_tangential_fad = boundary_layer_extraction.evaluate_pressure_gradient_tangential(soln_of_total_sampling,soln_grad_of_total_sampling)*ref_density*ref_speed*ref_speed/ref_length;
    FadType wall_shear_stress_fad   = boundary_layer_extraction.evaluate_wall_shear_stress(soln_of_total_sampling,soln_grad_of_total_sampling)*ref_density*ref_speed*ref_speed;
    FadType kinematic_viscosity_fad = boundary_layer_extraction.evaluate_kinematic_viscosity(soln_of_total_sampling)*ref_kinematic_viscosity;

    std::vector<FadType> Phi_pp_fad;
    std::vector<FadType> S_pp_fad;
    Phi_pp_fad.resize(numb_of_omega);
    S_pp_fad.resize(numb_of_omega);

    for (int i=0;i<numb_of_omega;++i){
        // Todo: create a vector for omega?
        const real omega_of_sampling = omega_min+i*d_omega;
        Phi_pp_fad[i] = wall_pressure_PSD(omega_of_sampling,
                                          speed_free_stream,
                                          density_free_stream,
                                          edge_velocity,
                                          boundary_layer_thickness,
                                          maximum_shear_stress,
                                          displacement_thickness_fad,
                                          momentum_thickness_fad,
                                          friction_velocity_fad,
                                          wall_shear_stress_fad,
                                          pressure_gradient_tangential_fad,
                                          kinematic_viscosity_fad);
        S_pp_fad[i] = acoustic_PSD(omega_of_sampling,U_c,Phi_pp_fad[i]);
    }

    output_wall_pressure_acoustic_spectrum_dat(Phi_pp_fad,S_pp_fad);

    FadType OASPL_fad = evaluate_overall_sound_pressure_level(S_pp_fad);

    // Todo: dIdW_int needs to be distributed vector to conduct multiplication of dIdW_int and dW_int_dW
    std::vector<real> dIdW_int;
    dIdW_int.resize(n_total_int_indep);
    std::vector<real> dIdW_grad_int;
    dIdW_grad_int.resize(3*n_total_int_indep);

    i_int_derivative = 0;
    for(int int_i=0;int_i<number_of_total_sampling;++int_i){
        for(int s=0;s<nstate;++s){
            dIdW_int[int_i*nstate+s] = OASPL_fad.dx(i_int_derivative++);
        }
    }
    AssertDimension(i_int_derivative, n_total_int_indep);
    for(int int_i=0;int_i<number_of_total_sampling;++int_i){
        for(int s=0;s<nstate;++s){
            for(int d=0;d<dim;++d){
                dIdW_grad_int[dim*n_total_int_indep+int_i*nstate+s] = OASPL_fad.dx(i_int_derivative++);
            }
        }
    }
    AssertDimension(i_int_derivative, n_total_int_indep+3*n_total_int_indep);

    //dW_int_dW.Tvmult(dIdw, dIdW_int);

    std::vector<real> dIdw(this->dg->dof_handler.n_dofs());
    for(long unsigned int col=0;col<dIdw.size();++col){
        for(int row=0;row<n_total_int_indep;++row){
            dIdw[col] += dIdW_int[row]*dW_int_dW[row][col];
        }
    }
    for(long unsigned int col=0;col<dIdw.size();++col){
        for(int row=0;row<3*n_total_int_indep;++row){
            dIdw[col] += dIdW_grad_int[row]*dW_grad_int_dW[row][col];
        }
    }

    this->current_functional_value = OASPL_fad.val();

    return this->current_functional_value;
}
////----------------------------------------------------------------
//template <int dim, int nstate, typename real, typename MeshType>
//template <typename real2>
//void AmietModelFunctional<dim,nstate,real,MeshType>
//::allocate_W_X(dealii::LinearAlgebra::distributed::Vector<real2> &solution_input,
//               dealii::LinearAlgebra::distributed::Vector<real2> &volume_nodes_input) const
//{
//    // allocating the vector for solution_input
//    dealii::IndexSet locally_owned_dofs_soln = dg->dof_handler.locally_owned_dofs();
//    solution_input.reinit(locally_owned_dofs_soln, MPI_COMM_WORLD);
//
//    // allocating the vector for volume_nodes_input
//    dealii::IndexSet locally_owned_dofs_vol = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
//    dealii::IndexSet locally_relevant_dofs, ghost_dofs;
//    dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);
//    ghost_dofs = locally_relevant_dofs;
//    ghost_dofs.subtract_set(locally_owned_dofs_vol);
//    volume_nodes_input.reinit(locally_owned_dofs_vol, ghost_dofs, MPI_COMM_WORLD);
//}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD(const real omega,
                    const real speed_free_stream,
                    const real density_free_stream,
                    const real edge_velocity,
                    const real boundary_layer_thickness,
                    const real maximum_shear_stress,
                    const real2 displacement_thickness,
                    const real2 momentum_thickness,
                    const real2 friction_velocity,
                    const real2 wall_shear_stress,
                    const real2 pressure_gradient_tangential,
                    const real2 kinematic_viscosity) const
{
    switch(this->wall_pressure_spectral_model_type) {
        case Wall_Pressure_Spectral_Model_types::Goody : 
            return this->wall_pressure_PSD_Goody(omega,
                                                 edge_velocity,
                                                 boundary_layer_thickness,
                                                 friction_velocity,
                                                 wall_shear_stress,
                                                 kinematic_viscosity);
            break;
        case Wall_Pressure_Spectral_Model_types::Rozenberg : 
            return this->wall_pressure_PSD_Rozenburg(omega,
                                                     edge_velocity,
                                                     boundary_layer_thickness,
                                                     maximum_shear_stress,
                                                     displacement_thickness,
                                                     momentum_thickness,
                                                     friction_velocity,
                                                     wall_shear_stress,
                                                     pressure_gradient_tangential,
                                                     kinematic_viscosity);
            break;
        case Wall_Pressure_Spectral_Model_types::Kamruzzaman : 
            return this->wall_pressure_PSD_Kamruzzaman(omega,
                                                       speed_free_stream,
                                                       density_free_stream,
                                                       edge_velocity,
                                                       displacement_thickness,
                                                       momentum_thickness,
                                                       friction_velocity,
                                                       wall_shear_stress,
                                                       pressure_gradient_tangential,
                                                       kinematic_viscosity);
            break;
        default: 
            break;
    }
    std::cout << "ERROR: Fail to determine wall pressure spectrum model type for Amiet's model..." << std::endl;
    std::abort();
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD_Goody(const real omega,
                          const real edge_velocity,
                          const real boundary_layer_thickness,
                          const real2 friction_velocity,
                          const real2 wall_shear_stress,
                          const real2 kinematic_viscosity) const
{
    // Goody's model for wall pressure spectrum
    const real a = 3.0;
    const real b = 2.0;
    const real c = 0.75;
    const real d = 0.5;
    const real e = 3.7;
    const real f = 1.1;
    const real g = -0.57;
    const real h = 7.0;
    const real i = 1.0;

    const real2 Phi_star = wall_shear_stress*wall_shear_stress*boundary_layer_thickness/edge_velocity;
    //const real2 R_T = (boundary_layer_thickness/edge_velocity)/(kinematic_viscosity/(friction_velocity*friction_velocity));
    const real2 R_T = evaluate_time_scale_ratio(boundary_layer_thickness,edge_velocity,friction_velocity,kinematic_viscosity);
    const real2 omega_star = omega*boundary_layer_thickness/edge_velocity;

    real2 Phi_pp = Phi_star*a*pow(omega_star,b)/(pow(i*pow(omega_star,c)+d,e)+pow(f*pow(R_T,g)*omega_star,h));

    return Phi_pp;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD_Rozenburg(const real omega,
                              const real edge_velocity,
                              const real boundary_layer_thickness,
                              const real maximum_shear_stress,
                              const real2 displacement_thickness,
                              const real2 momentum_thickness,
                              const real2 friction_velocity,
                              const real2 wall_shear_stress,
                              const real2 pressure_gradient_tangential,
                              const real2 kinematic_viscosity) const
{
    // Rozenburg's model for wall pressure spectrum
    const real2 Pi = evaluate_cole_wake_parameter(momentum_thickness,wall_shear_stress,pressure_gradient_tangential);
    const real2 Delta_star = evaluate_zagarola_smits_parameter(boundary_layer_thickness,displacement_thickness);
    const real2 beta_c = evaluate_clauser_equilibrium_parameter(momentum_thickness,wall_shear_stress,pressure_gradient_tangential);
    const real b = 2.0;
    const real c = 0.75;
    const real2 e = 3.7+1.5*beta_c;
    const real2 d = 4.76*pow(1.4/Delta_star,0.75)*(0.375*e-1.0);
    const real2 a = 2.82*Delta_star*Delta_star*pow(6.13*pow(Delta_star,-0.75)+d,e)*(4.2*Pi/Delta_star+1.0);
    const real f = 8.8;
    const real g = -0.57;
    const real i = 4.76;
    //const real2 R_T = (boundary_layer_thickness/edge_velocity)/(kinematic_viscosity/(friction_velocity*friction_velocity));
    const real2 R_T = evaluate_time_scale_ratio(boundary_layer_thickness,edge_velocity,friction_velocity,kinematic_viscosity);

    const real2 const_l = 3.0;
    const real2 const_r = 19.0/sqrt(R_T);
#if const_l<const_r
    const real h = 3.0+7.0;
#else
    const real2 h =19.0/sqrt(R_T)+7.0 ;
#endif
    (void) const_l;
    (void) const_r;
  
    const real2 Phi_star = maximum_shear_stress*maximum_shear_stress*displacement_thickness/edge_velocity;
    const real2 omega_star = omega*displacement_thickness/edge_velocity;

    real2 Phi_pp = Phi_star*a*pow(omega_star,b)/(pow(i*pow(omega_star,c)+d,e)+pow(f*pow(R_T,g)*omega_star,h));

    return Phi_pp;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::wall_pressure_PSD_Kamruzzaman(const real omega,
                                const real speed_free_stream,
                                const real density_free_stream,
                                const real edge_velocity,
                                const real2 displacement_thickness,
                                const real2 momentum_thickness,
                                const real2 friction_velocity,
                                const real2 wall_shear_stress,
                                const real2 pressure_gradient_tangential,
                                const real2 kinematic_viscosity) const
{
    // Kamruzzaman's model for wall pressure spectrum
    const real2 Pi = evaluate_cole_wake_parameter(momentum_thickness,wall_shear_stress,pressure_gradient_tangential);
    const real2 beta_c = evaluate_clauser_equilibrium_parameter(momentum_thickness,wall_shear_stress,pressure_gradient_tangential);
    const real2 G = 6.1*sqrt(beta_c+1.81)-1.7;
    const real2 C_f = wall_shear_stress/(0.5*density_free_stream*speed_free_stream*speed_free_stream);
    const real2 lambda = sqrt(2.0/C_f);
    const real2 H = 1.0-G/lambda; 
    const real2 m = 0.5*pow(H/1.31,0.3);
    const real2 a = 0.45*(1.75*pow(Pi*Pi*beta_c*beta_c,m)+15.0);
    const real b = 2.0;
    const real c = 1.637;
    const real d = 0.27;
    const real e = 2.47;
    const real f = pow(1.15,-2.0/7.0);
    const real g = -2.0/7.0;
    const real h = 7.0;
    const real i = 1.0;

    // Note: evaluate_time_scale_ratio can not be used due to the type of 1st arguement is real2
    const real2 R_T = (displacement_thickness/edge_velocity)/(kinematic_viscosity/(friction_velocity*friction_velocity));
    const real2 Phi_star = wall_shear_stress*wall_shear_stress*displacement_thickness/edge_velocity;
    const real2 omega_star = omega*displacement_thickness/edge_velocity;

    real2 Phi_pp = Phi_star*a*pow(omega_star,b)/(pow(i*pow(omega_star,c)+d,e)+pow(f*pow(R_T,g)*omega_star,h));

    return Phi_pp;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_time_scale_ratio(const real boundary_layer_thickness,
                            const real edge_velocity,
                            const real2 friction_velocity,
                            const real2 kinematic_viscosity) const
{
    return (boundary_layer_thickness/edge_velocity)/(kinematic_viscosity/(friction_velocity*friction_velocity));
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_clauser_equilibrium_parameter(const real2 momentum_thickness,
                                         const real2 wall_shear_stress,
                                         const real2 pressure_gradient_tangential) const
{
    return momentum_thickness/wall_shear_stress*pressure_gradient_tangential;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_cole_wake_parameter(const real2 momentum_thickness,
                               const real2 wall_shear_stress,
                               const real2 pressure_gradient_tangential) const
{
    const real2 beta_c = evaluate_clauser_equilibrium_parameter(momentum_thickness,
                                                                wall_shear_stress,
                                                                pressure_gradient_tangential);
    return 0.8*pow(beta_c+0.5,3.0/4.0);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_zagarola_smits_parameter(const real boundary_layer_thickness,
                                    const real2 displacement_thickness) const
{
    return boundary_layer_thickness/displacement_thickness;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real AmietModelFunctional<dim,nstate,real,MeshType>
::spanwise_correlation_length(const real omega,
                              const real U_c) const
{
    // Corcos model for spanwise correlation length
    const real b_c = 1.47;
    return b_c*U_c/omega;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::E(const std::complex<real> z) const
{
    return (1.0+imag_unit)/2.0*Faddeeva::erf(sqrt(-imag_unit*z));
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::E_star(const std::complex<real> z) const
{
    return (1.0-imag_unit)/2.0*Faddeeva::erf(sqrt(imag_unit*z));
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::ES_star(const std::complex<real> z) const
{
    std::complex<real> ES_star;
    if (z.real()==0.0 && z.imag()==0.0){
        ES_star = sqrt(2.0/pi);
    } else {
        ES_star = E_star(z)/sqrt(z);
    }
    if (z.real()<0.0 && z.imag()>=0.0) {
        ES_star *= -1.0;
    }
    return ES_star;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::radiation_integral_trailing_edge_main (const real B,
                                         const real C,
                                         const real mu_bar,
                                         const real S0,
                                         const real kappa_bar_prime,
                                         const std::complex<real> A1_prime,
                                         const bool is_supercritical) const
{
    std::complex<real> radiation_integral_trailing_edge_main;
    
    if (is_supercritical) {
        radiation_integral_trailing_edge_main 
            = -exp(2.0*imag_unit*C)/(imag_unit*C)*((1.0+imag_unit)*exp(-2.0*imag_unit*C)*sqrt(2.0*B)*ES_star(2.0*(B-C))-(1.0+imag_unit)*E_star(2.0*B)+1.0-exp(-2.0*imag_unit*C));
    } else {
        radiation_integral_trailing_edge_main 
            = -exp(2.0*imag_unit*C)/(imag_unit*C)*(exp(-2.0*imag_unit*C)*sqrt(2.0*A1_prime)*(1.0+imag_unit)*ES_star(2.0*(mu_bar*observer_coord_ref[0]/S0-imag_unit*kappa_bar_prime))-Faddeeva::erf(sqrt(2.0*imag_unit*A1_prime))+1.0);
    }
    
    return radiation_integral_trailing_edge_main;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::radiation_integral_trailing_edge_back (const real C,
                                         const real D,
                                         const real kappa_bar,
                                         const real kappa_bar_prime,
                                         const real K_bar,
                                         const real mu_bar,
                                         const real S0,
                                         const real mach_inf,
                                         const std::complex<real> A_prime,
                                         const std::complex<real> G,
                                         const std::complex<real> D_prime,
                                         const std::complex<real> H,
                                         const std::complex<real> H_prime,
                                         const bool is_supercritical) const 
{
    std::complex<real> radiation_integral_trailing_edge_back;

    if (is_supercritical) {
      radiation_integral_trailing_edge_back
          = H*(pow(exp(4.0*imag_unit*kappa_bar)*(1.0-(1.0+imag_unit)*E_star(4.0*kappa_bar)),C)-exp(2.0*imag_unit*D)+imag_unit*(D+K_bar+mach_inf*mu_bar-kappa_bar)*G);
    } else {
      radiation_integral_trailing_edge_back
          = exp(-2.0*imag_unit*D_prime)/D_prime*H_prime*(A_prime*(exp(2.0*imag_unit*D_prime)*(1.0-Faddeeva::erf(sqrt(4.0*kappa_bar_prime)))-1.0)+2.0*sqrt(2.0*kappa_bar_prime)*(K_bar+(mach_inf-observer_coord_ref[0]/S0)*mu_bar)*ES_star(-2.0*D_prime));
    }
    
    return radiation_integral_trailing_edge_back;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::complex<real> AmietModelFunctional<dim,nstate,real,MeshType>
::radiation_integral_trailing_edge (const real omega) const
{
    const real k = omega/sound_inf;
    const real K = k/mach_inf;
    const real K_bar = K*b;
    const real K1_bar = K_bar*alpha;
    const real K2_bar = k*observer_coord_ref[1]/S0*b;
    const real mu_bar = K_bar*mach_inf/beta_sqr;
    const real kappa_bar = sqrt(mu_bar*mu_bar-K2_bar*K2_bar/beta_sqr);
    const real kappa_bar_prime = sqrt(K2_bar*K2_bar/beta_sqr-mu_bar*mu_bar);
    const real epsilon = pow(1.0+1.0/(4.0*mu_bar),-1.0/2.0);
    
    const real Theta = sqrt((K1_bar+mach_inf*mu_bar+kappa_bar)/(K_bar+mach_inf*mu_bar+kappa_bar));
    const std::complex<real> A_prime = K_bar+mach_inf*mu_bar-kappa_bar_prime*imag_unit;
    const std::complex<real> A1_prime = K1_bar+mach_inf*mu_bar-kappa_bar_prime*imag_unit;
    const std::complex<real> Theta_prime = sqrt(A1_prime/A_prime);
    
    const real B = K1_bar+mach_inf*mu_bar+kappa_bar;
    const real C = K1_bar-mu_bar*(observer_coord_ref[0]/S0-mach_inf);
    const real D = kappa_bar-mu_bar*observer_coord_ref[0]/S0;
    const std::complex<real> D_prime = mu_bar*observer_coord_ref[0]/S0-kappa_bar_prime*imag_unit;
    const std::complex<real> 
        G = (1.0+epsilon)*exp(imag_unit*(2.0*kappa_bar+D))*sin(D-2.0*kappa_bar)/(D-2.0*kappa_bar)
            +(1.0-epsilon)*exp(imag_unit*(-2.0*kappa_bar+D))*sin(D+2.0*kappa_bar)/(D+2.0*kappa_bar)
            +(1.0+epsilon)*(1.0-imag_unit)/(2.0*(D-2.0*kappa_bar))*exp(4.0*kappa_bar*imag_unit)*E_star(4.0*kappa_bar)
            -(1.0-epsilon)*(1.0+imag_unit)/(2.0*(D+2.0*kappa_bar))*exp(-4.0*kappa_bar*imag_unit)*E(4.0*kappa_bar)
            +exp(2.0*D*imag_unit)/sqrt(2.0)*sqrt((2.0*kappa_bar)/D)*E_star(2.0*D)*((1.0+imag_unit)*(1.0-epsilon)/(D+2.0*kappa_bar)-(1.0-imag_unit)*(1.0+epsilon)/(D-2.0*kappa_bar));
    const std::complex<real> 
        H = (1.0+imag_unit)*exp(-4.0*kappa_bar*imag_unit)*(1.0-Theta*Theta)/(2.0*sqrt(pi)*(alpha-1.0)*K_bar*sqrt(B));
    const std::complex<real>
        H_prime = (1.0+imag_unit)*(1.0-Theta_prime*Theta_prime)/(2.0*sqrt(pi)*(alpha-1.0)*K_bar*sqrt(A1_prime));

    const bool is_supercritical = K2_bar*K2_bar < mu_bar*mu_bar*beta_sqr;

    std::complex<real> RI_TE_main;
    std::complex<real> RI_TE_back;
    switch(this->acoustic_contribution_type) {
        case Acoustic_Contribution_types::main : 
            RI_TE_main = radiation_integral_trailing_edge_main(B,C,mu_bar,S0,kappa_bar_prime,A1_prime,is_supercritical);
            return RI_TE_main;
            break;
        case Acoustic_Contribution_types::back : 
            RI_TE_back = radiation_integral_trailing_edge_back(C,D,kappa_bar,kappa_bar_prime,K_bar,mu_bar,S0,mach_inf,A_prime,G,D_prime,H,H_prime,is_supercritical);
            return RI_TE_back;
            break;
        case Acoustic_Contribution_types::main_and_back : 
            RI_TE_main = radiation_integral_trailing_edge_main(B,C,mu_bar,S0,kappa_bar_prime,A1_prime,is_supercritical);
            RI_TE_back = radiation_integral_trailing_edge_back(C,D,kappa_bar,kappa_bar_prime,K_bar,mu_bar,S0,mach_inf,A_prime,G,D_prime,H,H_prime,is_supercritical);
            return RI_TE_main+RI_TE_back;
            break;
        default: 
            break;
    }
    std::cout << "ERROR: Fail to determine contribution type for radiation integral..." << std::endl;
    std::abort();
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::acoustic_PSD(const real omega,
               const real U_c,
               const real2 Phi_pp_of_sampling) const
{
    const std::complex<real> radiation_integral = radiation_integral_trailing_edge(omega);
    const real l_y = spanwise_correlation_length(omega,U_c);
 
    real2 S_pp_of_sampling;

    S_pp_of_sampling = pow(omega*observer_coord_ref[2]*b/(2.0*pi*sound_inf*S0*S0),2.0)*2.0*span_length*pow(abs(radiation_integral),2.0)*Phi_pp_of_sampling*l_y;

    return S_pp_of_sampling;
}
////----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//void AmietModelFunctional<dim,nstate,real,MeshType>
//::evaluate_wall_pressure_acoustic_spectrum()
//{   
//    for (int i=0;i<numb_of_omega;++i){
//        const real omega_of_sampling = omega_min+i*d_omega;
//        Phi_pp[i] = wall_pressure_PSD(omega_of_sampling);
//        S_pp[i] = acoustic_PSD(omega_of_sampling,Phi_pp[i]);
//    }
//}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 AmietModelFunctional<dim,nstate,real,MeshType>
::evaluate_overall_sound_pressure_level(const std::vector<real2> S_pp)
{   
    real2 overall_PSD = 0.0;
    for (int i=0;i<numb_of_omega;++i){
        overall_PSD += S_pp[i]*d_omega;
    }
    real2 overall_SPL = 10.0*log10(8.0*pi*overall_PSD/(2e-5*2e-5));
    return overall_SPL;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
void AmietModelFunctional<dim,nstate,real,MeshType>
::output_wall_pressure_acoustic_spectrum_dat(const std::vector<real2> &Phi_pp,
                                             const std::vector<real2> &S_pp)
{   
    std::ofstream outfile_S_pp_Phi_pp;
    outfile_S_pp_Phi_pp.open("S_pp_and_Phi_pp.dat");
    for(int i=0;i<numb_of_omega;++i){
        const real omega_of_sampling = omega_min+i*d_omega;
        if constexpr(std::is_same<real2,real>::value){
            outfile_S_pp_Phi_pp << omega_of_sampling/(2.0*pi) << "\t\t" 
                                << 10.0*log10(8.0*pi*S_pp[i]/(2e-5*2e-5)) << "\t\t" 
                                << 10.0*log10(2.0*pi*Phi_pp[i]/(2e-5*2e-5)) << "\n";
        }
        else if constexpr(std::is_same<real2,FadType>::value){
            outfile_S_pp_Phi_pp << omega_of_sampling/(2.0*pi) << "\t\t" 
                                << 10.0*log10(8.0*pi*S_pp[i].val()/(2e-5*2e-5)) << "\t\t" 
                                << 10.0*log10(2.0*pi*Phi_pp[i].val()/(2e-5*2e-5)) << "\n";
        }
    }
    outfile_S_pp_Phi_pp.close();
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- AmietModelFunctional
#if PHILIP_DIM!=1
template class AmietModelFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AmietModelFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace