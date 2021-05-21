#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/msh_out.h"
#include "grid_refinement/size_field.h"
#include "grid_refinement/reconstruct_poly.h"
#include "grid_refinement/field.h"

#include "grid_refinement_continuous.h"

namespace PHiLiP {

namespace GridRefinement {

// Function class to rezero the solution on grid read
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

template <int dim, int nstate, typename real, typename MeshType>
GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous(
    PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) :
        GridRefinement_Continuous<dim,nstate,real,MeshType>(
            gr_param_input,
            adj_input,
            adj_input->functional, 
            adj_input->dg, 
            physics_input){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous(
    PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input) : 
        GridRefinement_Continuous<dim,nstate,real,MeshType>(
            gr_param_input, 
            nullptr, 
            functional_input, 
            dg_input, 
            physics_input){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous(
    PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) : 
        GridRefinement_Continuous<dim,nstate,real,MeshType>(
            gr_param_input, 
            nullptr, 
            nullptr, 
            dg_input, 
            physics_input){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous(
    PHiLiP::Parameters::GridRefinementParam                gr_param_input,
    // PHiLiP::Parameters::AllParameters const *const param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg_input) :
        GridRefinement_Continuous<dim,nstate,real,MeshType>(
            gr_param_input, 
            nullptr, 
            nullptr, 
            dg_input, 
            nullptr){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinement_Continuous<dim,nstate,real,MeshType>::GridRefinement_Continuous(
    PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >    adj_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input) :
        GridRefinementBase<dim,nstate,real,MeshType>(
            gr_param_input,
            adj_input,
            functional_input,
            dg_input,
            physics_input)
{
    std::cout << "Initializing grid_refinement with anisotropic = " << this->grid_refinement_param.anisotropic << std::endl;

    // sets the Field to default to either anisotropic or isotropic field
    if(this->grid_refinement_param.anisotropic){
        h_field = std::make_unique<FieldAnisotropic<dim,real>>();
    }else{
        h_field = std::make_unique<FieldIsotropic<dim,real>>();
    }

    // set the initial complexity
    complexity_initial = current_complexity();

    // copy the complexity vector (if any) from the parameters
    complexity_vector = this->grid_refinement_param.complexity_vector;

    // adds first element if none hav been yet
    complexity_vector.push_back(
        complexity_initial
        * this->grid_refinement_param.complexity_scale 
        + this->grid_refinement_param.complexity_add);

    // set the intial target
    target_complexity();
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // store the previous solution space

    // compute the necessary size fields
    field();

    // generate a new grid
    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    // reinitialize the dg object with new coarse triangulation
    this->dg->reinit();

    // interpolate the solution from the previous solution space
    this->dg->allocate_system();

    // zeroing the solution
    std::shared_ptr<dealii::Function<dim>> initial_conditions = 
        std::make_shared<InitialConditions<dim,nstate>>();
    
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(this->dg->locally_owned_dofs, MPI_COMM_WORLD);
    const auto mapping = *(this->dg->high_order_grid->mapping_fe_field);
    dealii::VectorTools::interpolate(mapping, this->dg->dof_handler, *initial_conditions, solution_no_ghost);
    this->dg->solution = solution_no_ghost;

    // increase the count
    this->iteration++;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_h()
{
    using OutputType = PHiLiP::Parameters::GridRefinementParam::OutputType;
    OutputType output_type = this->grid_refinement_param.output_type;

    // selecting the output type
    if(output_type == OutputType::gmsh_out){
        refine_grid_gmsh();
    }else if(output_type == OutputType::msh_out){
        refine_grid_msh();
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_p()
{
    // physical grid stays the same, apply the update to the p_field
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            cell->set_future_fe_index(round(p_field[cell->active_cell_index()]));
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_hp()
{
    // NOT IMPLEMENTED
    assert(0);

    // make a copy of the old grid and build a P1 continuous solution averaged at each of the nodes
    // new P will be the weighted average of the integral over the new cell
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_gmsh()
{
    const int iproc = dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator);
    
    // now outputting this new field
    std::string write_posname = "grid-" + 
                                dealii::Utilities::int_to_string(this->iteration, 4) + "." + 
                                dealii::Utilities::int_to_string(iproc, 4) + ".pos";
    std::ofstream outpos(write_posname);

    // check for anisotropy
    if(this->grid_refinement_param.anisotropic){
        // polynomial order from max
        const int poly_degree = this->dg->get_max_fe_degree();

        // using an anisotropic BAMG with merge
        GmshOut<dim,real>::write_pos_anisotropic(
            *(this->tria),
            this->h_field->get_inverse_quadratic_metric_vector(),
            outpos,
            poly_degree);
    }else{
        // using a frontal approach to quad generation
        GmshOut<dim,real>::write_pos(
            *(this->tria),
            this->h_field->get_scale_vector_dealii(),
            outpos);
    }

    // writing the geo file on the 1st processor and running
    std::string output_name = "grid-" + 
                              dealii::Utilities::int_to_string(this->iteration, 4) + ".msh";
    if(iproc == 0){
        // generating a vector of pos file names
        std::vector<std::string> posname_vec;
        for(unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator); ++iproc)
            posname_vec.push_back("grid-" + 
                                  dealii::Utilities::int_to_string(this->iteration, 4) + "." + 
                                  dealii::Utilities::int_to_string(iproc, 4) + ".pos");

        std::string write_geoname = "grid-" + 
                                    dealii::Utilities::int_to_string(this->iteration, 4) + ".geo";
        std::ofstream outgeo(write_geoname);
        
        // check for anisotropy
        if(this->grid_refinement_param.anisotropic){
            // using an anisotropic BAMG with merge
            GmshOut<dim,real>::write_geo_anisotropic(
                posname_vec,
                outgeo);
        }else{
            // using a frontal approach to quad generation
            GmshOut<dim,real>::write_geo(
                posname_vec,
                outgeo);
        }

        // run gmsh with check for non-availibility
        if(!GmshOut<dim,real>::call_gmsh(write_geoname, output_name))
            return;
    }

    // barrier
    MPI_Barrier(this->mpi_communicator);
    
    // loading the mesh on all processors
    this->tria->clear();
    dealii::GridIn<dim> gridin;
    gridin.attach_triangulation(*(this->tria));
    std::ifstream f(output_name);
    gridin.read_msh(f);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_msh()
{
    // geting the data output type
    using OutputDataType = PHiLiP::Parameters::GridRefinementParam::OutputDataType;
    OutputDataType output_data_type = this->grid_refinement_param.output_data_type;

    // storage type for format of data vector
    using StorageType = PHiLiP::GridRefinement::StorageType;
    StorageType storage_type = StorageType::element;

    // file name and output file stream
    std::string write_msh_name = "grid-a" + 
                                 dealii::Utilities::int_to_string(this->iteration, 4) + ".msh";
    
    std::ofstream out_msh(write_msh_name);

    // setting up output handler
    PHiLiP::GridRefinement::MshOut<dim,real> msh_out(this->dg->dof_handler);

    // writing the field to output (if uncommented)
    // std::cout << *(this->h_field) << std::endl;

    if(output_data_type == OutputDataType::size_field){
        // outputting the h_field size (no orientation), single value
        std::cout << "Writing the size_field to .msh file." << std::endl;

        // adding data to output
        msh_out.add_data_vector(this->h_field->get_scale_vector(), 
                                storage_type, 
                                "size_field");
        
    }else if(output_data_type == OutputDataType::frame_field){
        // outputting the h_field frame vectors (d, 1xd vectors)
        std::cout << "Writing the frame_field to .msh file." << std::endl;

        // adding data to output
        for(unsigned int j = 0; j < dim; ++j)
            msh_out.add_data_vector(this->h_field->get_axis_vector(j),
                                    storage_type,
                                    "frame_field_" + std::to_string(j));


    }else if(output_data_type == OutputDataType::metric_field){
        // outputting the h_field using metric representation (dxd matrix)
        std::cout << "Writing the metric_field to .msh file." << std::endl;

        // adding data to output
        msh_out.add_data_vector(this->h_field->get_inverse_metric_vector(),
                                storage_type,
                                "Vinv");

        // msh_out.add_data_vector(this->h_field->get_metric_vector(),
        //                         storage_type,
        //                         "metric_field");

        // // for testing
        // std::vector<dealii::SymmetricTensor<2,dim,real>> quadratic_metric_sym = this->h_field->get_quadratic_metric_vector();
        // std::vector<dealii::Tensor<2,dim,real>> quadratic_metric;
        
        // quadratic_metric.reserve(quadratic_metric_sym.size());
        // for(auto metric: quadratic_metric_sym)
        //     quadratic_metric.push_back(metric);

        // msh_out.add_data_vector(quadratic_metric,
        //                         storage_type,
        //                         "metric_field_sym");

    }

    // writing the msh file
    msh_out.write_msh(out_msh);

    // full cycle-not yet implemented
    std::cout << ".msh file written. (" << "/" << write_msh_name << ")" << std::endl;

    // allow to continue only if it will exit immediately afterwards
    if(this->grid_refinement_param.exit_after_refine)
    {
        std::cout << "refine_grid_msh: read not supported, proceeding to results." << std::endl;
    }else{
        std::cout << "refine_grid_msh: read not supported, use option exit_after_refine to stop after .msh write." << std::endl;
        throw;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    using ErrorIndicatorEnum = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // updating the target complexity for this iteration
    target_complexity();

    // compute the necessary size fields
    if(refinement_type == RefinementTypeEnum::h){
        // mesh size and shape
        if(this->error_indicator_type == ErrorIndicatorEnum::error_based){
            field_h_error();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::hessian_based){
            field_h_hessian();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::residual_based){
            field_h_residual();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::adjoint_based){
            field_h_adjoint();
        }
    }else if(refinement_type == RefinementTypeEnum::p){
        // polynomial order distribution
        if(this->error_indicator_type == ErrorIndicatorEnum::error_based){
            field_p_error();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::hessian_based){
            field_p_hessian();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::residual_based){
            field_p_residual();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::adjoint_based){
            field_p_adjoint();
        }
    }else if(refinement_type == RefinementTypeEnum::hp){
        // combined refinement methods
        if(this->error_indicator_type == ErrorIndicatorEnum::error_based){
            field_hp_error();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::hessian_based){
            field_hp_hessian();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::residual_based){
            field_hp_residual();
        }else if(this->error_indicator_type == ErrorIndicatorEnum::adjoint_based){
            field_hp_adjoint();
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::target_complexity()
{
    // if the complexity vector needs to be expanded
    while(complexity_vector.size() <= this->iteration)
        complexity_vector.push_back(
            complexity_vector.back() 
          * this->grid_refinement_param.complexity_scale 
          + this->grid_refinement_param.complexity_add);

    // copy the current iteration into the complexity target
    complexity_target = complexity_vector[this->iteration];

    std::cout << "Complexity target = " << complexity_target << std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
real GridRefinement_Continuous<dim,nstate,real,MeshType>::current_complexity()
{
    real complexity_sum;

    // two possible cases
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        // case 1: uniform p-order, complexity relates to total dof
        unsigned int poly_degree = this->dg->get_min_fe_degree();
        return pow(poly_degree+1, dim) * this->tria->n_global_active_cells(); 
    }else{
        // case 2: non-uniform p-order, complexity related to the local sizes
        for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
            if(cell->is_locally_owned())
                complexity_sum += pow(cell->active_fe_index()+1, dim);
    }

    return dealii::Utilities::MPI::sum(complexity_sum, MPI_COMM_WORLD);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_h()
{
    // gets the current size and copy it into field_h
    // for isotropic, sets the size to be the h = volume ^ (1/dim)
    this->h_field->reinit(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            this->h_field->set_scale(cell->active_cell_index(), pow(cell->measure(), 1.0/dim));
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p()
{
    // gets the current polynomiala distribution and copies it into field_p
    p_field.reinit(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            this->p_field[cell->active_cell_index()] = cell->active_fe_index();
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_h_error()
{
    real q = 2.0;

    dealii::Vector<real> B(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // getting the central coordinate as average of vertices
        dealii::Point<dim,real> center_point = cell->center();

        // evaluating the Hessian at this point (using default state)
        dealii::SymmetricTensor<2,dim,real> hessian = 
            this->physics->manufactured_solution_function->hessian(center_point);

        // assuming 2D for now
        // TODO: check generalization of this for different dimensions with eigenvalues
        if(dim == 2)
            B[cell->active_cell_index()] = 
                pow(abs(hessian[0][0]*hessian[1][1] - hessian[0][1]*hessian[1][0]), q/2);
    }

    // checking if the polynomial order is uniform
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        unsigned int poly_degree = this->dg->get_min_fe_degree();

        // building error based on exact hessian
        SizeField<dim,real>::isotropic_uniform(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            this->h_field,
            poly_degree);

    }else{
        // call the non-uniform hp-version without the p-update
        GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p();

        // mapping
        const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));

        SizeField<dim,real>::isotropic_h(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags,
            this->h_field,
            this->p_field);

    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_p_error()
{
    // NOT IMPLEMENTED
    assert(0);
}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_hp_error()
{
    // NOT IMPLEMENTED
    assert(0);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_h_hessian()
{
    // beginning h_field computation
    std::cout << "Beggining field_h() computation" << std::endl;

    // mapping
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));

    // using p+1 reconstruction
    const unsigned int rel_order = 1;

    // construct object to reconstruct the derivatives onto A
    ReconstructPoly<dim,nstate,real> reconstruct_poly(
        this->dg->dof_handler,
        mapping_collection,
        this->dg->fe_collection,
        this->dg->volume_quadrature_collection,
        this->volume_update_flags);

    // constructing the largest directional derivatives
    reconstruct_poly.reconstruct_directional_derivative(
        this->dg->solution,
        rel_order);
    // reconstruct_poly.reconstruct_manufactured_derivative(
    //     this->physics->manufactured_solution_function,
    //     rel_order);

    // if anisotropic, setting the cell anisotropy
    if(this->grid_refinement_param.anisotropic){
        std::cout << "Setting cell anisotropy" << std::endl;

        // builds the anisotropy from Dolejsi's anisotropic ellipse size
        this->h_field->set_anisotropy(
            this->dg->dof_handler,
            reconstruct_poly.derivative_value,
            reconstruct_poly.derivative_direction,
            rel_order);

        std::cout << "Applying anisotropy limits: \\rho = [" << 
            this->grid_refinement_param.anisotropic_ratio_min << ", " <<
            this->grid_refinement_param.anisotropic_ratio_max << "]" << std::endl;

        this->h_field->apply_anisotropic_limit(
            this->grid_refinement_param.anisotropic_ratio_min,
            this->grid_refinement_param.anisotropic_ratio_max);

    }

    // vector to store the results for local scaling parameter
    dealii::Vector<real> B(this->tria->n_active_cells());

    // looping over the vector and taking the product of the eigenvalues as the size measure
    for(auto cell = this->dg->dof_handler.begin_active(); cell < this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned()){
            B[cell->active_cell_index()] = 1.0;
            for(unsigned int d = 0; d < dim; ++d)
                B[cell->active_cell_index()] *= reconstruct_poly.derivative_value[cell->active_cell_index()][d];
        }

    // setting the current p-field

    // TODO: perform the call to calculate the continuous size field

    // checking if the polynomial order is uniform
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        unsigned int poly_degree = this->dg->get_min_fe_degree();

        SizeField<dim,real>::isotropic_uniform(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            this->h_field,
            poly_degree);

    }else{
        // the case of non-uniform p
        GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p();

        SizeField<dim,real>::isotropic_h(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags,
            this->h_field,
            this->p_field);

    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_p_hessian()
{
    // NOT IMPLEMENTED
    assert(0);
}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_hp_hessian()
{
    // NOT IMPLEMENTED
    assert(0);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_h_residual()
{
    // NOT IMPLEMENTED
    assert(0);
}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_p_residual()
{
    // NOT IMPLEMENTED
    assert(0);
}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_hp_residual()
{
    // NOT IMPLEMENTED
    assert(0);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_h_adjoint()
{
    // beginning h_field computation
    std::cout << "Beggining anisotropic field_h() computation" << std::endl;

    // mapping
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));

    // using p+1 reconstruction
    const unsigned int rel_order = 1;

    // construct object to reconstruct the derivatives onto A
    ReconstructPoly<dim,nstate,real> reconstruct_poly(
        this->dg->dof_handler,
        mapping_collection,
        this->dg->fe_collection,
        this->dg->volume_quadrature_collection,
        this->volume_update_flags);

    // constructing the largest directional derivatives
    reconstruct_poly.reconstruct_directional_derivative(
        this->dg->solution,
        rel_order);
    
    // if anisotropic, setting the cell anisotropy
    if(this->grid_refinement_param.anisotropic){
        std::cout << "Setting cell anisotropy" << std::endl;

        // builds the anisotropy from Dolejsi's anisotropic ellipse size
        this->h_field->set_anisotropy(
            this->dg->dof_handler,
            reconstruct_poly.derivative_value,
            reconstruct_poly.derivative_direction,
            rel_order);

        std::cout << "Applying anisotropy limits: \\rho = [" << 
            this->grid_refinement_param.anisotropic_ratio_min << ", " <<
            this->grid_refinement_param.anisotropic_ratio_max << "]" << std::endl;

        this->h_field->apply_anisotropic_limit(
            this->grid_refinement_param.anisotropic_ratio_min,
            this->grid_refinement_param.anisotropic_ratio_max);

    }

    // getting the DWR (cell-wise indicator)
    this->adjoint->fine_grid_adjoint();
    this->adjoint->dual_weighted_residual();
    dealii::Vector<real> dwr = this->adjoint->dual_weighted_residual_fine;
    this->adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double,MeshType>::AdjointStateEnum::coarse);

    // setting the current p-field and performing the size-field refinement step

    // checking if the polynomial order is uniform
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        unsigned int poly_degree = this->dg->get_min_fe_degree();

        this->get_current_field_h();
        SizeField<dim,real>::adjoint_uniform_balan(
            this->complexity_target,
            this->grid_refinement_param.r_max,
            this->grid_refinement_param.c_max,
            dwr,
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags,
            this->h_field,
            poly_degree);
        // SizeField<dim,real>::adjoint_h_equal(
        //     this->complexity_target,
        //     dwr,
        //     this->dg->dof_handler,
        //     mapping_collection,
        //     this->dg->fe_collection,
        //     this->dg->volume_quadrature_collection,
        //     this->volume_update_flags,
        //     this->h_field,
        //     poly_degree);

    }else{
        // the case of non-uniform p
        GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p();

        this->get_current_field_h();
        SizeField<dim,real>::adjoint_h_balan(
            this->complexity_target,
            this->grid_refinement_param.r_max,
            this->grid_refinement_param.c_max,
            dwr,
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags,
            this->h_field,
            this->p_field);

    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_p_adjoint()
{
    // NOT IMPLEMENTED
    assert(0);
}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field_hp_adjoint()
{
    // NOT IMPLEMENTED
    assert(0);
}

template <int dim, int nstate, typename real, typename MeshType>
std::vector< std::pair<dealii::Vector<real>, std::string> > GridRefinement_Continuous<dim,nstate,real,MeshType>::output_results_vtk_method()
{
    std::vector< std::pair<dealii::Vector<real>, std::string> > data_out_vector;

    // getting the current field sizes
    get_current_field_h();
    data_out_vector.push_back(
        std::make_pair(
            this->h_field->get_scale_vector_dealii(), 
            "h_field_curr"));

    get_current_field_p();
    data_out_vector.push_back(
        std::make_pair(
            p_field, 
            "p_field_curr"));

    // computing the (next) update to the fields
    field();
    data_out_vector.push_back(
        std::make_pair(
            this->h_field->get_scale_vector_dealii(), 
            "h_field_next"));

    data_out_vector.push_back(
        std::make_pair(
            p_field, 
            "p_field_next"));

    // if field is anisotropic
    if(this->grid_refinement_param.anisotropic){
        // also outputting the anisotropic ratio of each cell
        data_out_vector.push_back(
            std::make_pair(
                this->h_field->get_max_anisotropic_ratio_vector_dealii(), 
                "anisotropic_ratio_next"));
    
        // reconstructing the directional derivatives again
        // mapping
        const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));

        // using p+1 reconstruction
        const unsigned int rel_order = 1;

        // construct object to reconstruct the derivatives onto A
        ReconstructPoly<dim,nstate,real> reconstruct_poly(
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags);

        // constructing the largest directional derivatives
        reconstruct_poly.reconstruct_directional_derivative(
            this->dg->solution,
            rel_order);
        // reconstruct_poly.reconstruct_manufactured_derivative(
        //     this->physics->manufactured_solution_function,
        //     rel_order);

        // getting the derivative_values as a dealii vector (in order)
        for(unsigned int i = 0; i < dim; ++i)
            data_out_vector.push_back(
                std::make_pair(
                    reconstruct_poly.get_derivative_value_vector_dealii(i), 
                    "derivative_value_" + dealii::Utilities::int_to_string(i, 1)));

    }

    return data_out_vector;

}

// dealii::Triangulation<PHILIP_DIM>
template class GridRefinement_Continuous<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

// dealii::parallel::shared::Triangulation<PHILIP_DIM>
template class GridRefinement_Continuous<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM != 1
// dealii::parallel::distributed::Triangulation<PHILIP_DIM>
template class GridRefinement_Continuous<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace GridRefinement

} // namespace PHiLiP
