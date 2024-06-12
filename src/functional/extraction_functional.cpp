#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "physics/physics_factory.h"
#include "physics/model_factory.h"

#include "extraction_functional.hpp"

#include <memory>

namespace PHiLiP {

//================================================================
// Extraction functional
//================================================================
template <int dim,int nstate,typename real,typename MeshType>
ExtractionFunctional<dim,nstate,real,MeshType>
::ExtractionFunctional(
    std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
    const dealii::Point<dim,real> start_point_input,
    const int number_of_sampling_input)
    : Functional<dim,nstate,real,MeshType>(dg_input)
    , start_point(start_point_input)
    , number_of_sampling(number_of_sampling_input)
    , number_of_total_sampling(number_of_sampling+2)
    , navier_stokes_fad(dynamic_cast< Physics::NavierStokes<dim,dim+2,FadType> &>(*(PHiLiP::Physics::PhysicsFactory<dim,dim+2,FadType>::create_Physics(this->dg->all_parameters, Parameters::AllParameters::PartialDifferentialEquation::navier_stokes))))
{
    if (nstate==dim+3){
        rans_sa_neg_fad = std::dynamic_pointer_cast< Physics::ReynoldsAveragedNavierStokes_SAneg<dim,dim+3,FadType> >(PHiLiP::Physics::ModelFactory<dim,dim+3,FadType>::create_Model(this->dg->all_parameters)); 
    }

    this->evaluate_extraction_start_point_coord();
    //std::cout << "Original start_point of boundary layer extraction is " << start_point[0] << "," << start_point[1] << std::endl;
    //for (auto cell = this->dg->triangulation->begin_active(); cell != this->dg->triangulation->end(); ++cell) {
    //    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
    //        if (cell->face(face)->at_boundary()) {
    //            // Todo: need a more general condition
    //            if (std::abs(cell->face(face)->center()[0]-start_point[0])<=1e-3 && cell->face(face)->center()[1]>=0.0){
    //                start_point = cell->face(face)->center();
    //            }
    //        }
    //    }
    //}
    //std::cout << "Corrected start_point of boundary layer extraction is " << start_point[0] << "," << start_point[1] << std::endl;

    this->evaluate_extraction_start_point_normal_tangential_vector();
    //const auto extraction_cell = dealii::GridTools::find_active_cell_around_point(*(this->dg->triangulation),start_point);
    //if(extraction_cell->at_boundary()){
    //    std::cout << "Captured cell that extraction point belongs to..." << std::endl;
    //    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
    //        if (extraction_cell->face(face)->at_boundary()) {
    //            start_point_normal_vector = extraction_cell->face(face)->get_manifold().normal_vector(extraction_cell->face(face),start_point);
    //            start_point_normal_vector*= -1.0;
    //            dealii::Point<dim,real> start_point_neighbor = start_point;
    //            // Todo: need to change to the point on the same boundary line/surface
    //            start_point_neighbor[0] += 1e-3;
    //            start_point_tangential_vector = extraction_cell->face(face)->get_manifold().get_tangent_vector(start_point,start_point_neighbor);
    //            for(int d=0;d<dim;++d){
    //                start_point_tangential_vector[d] /= start_point_tangential_vector.norm();
    //            }
    //            std::cout << "Captured normal and tangential vector of extraction point..." << std::endl;
    //        }
    //    }
    //} else {
    //    std::cout << "ERROR: Fail to capture cell that extraction point belongs to..." << std::endl;
    //    std::abort();
    //}

    this->evaluate_extraction_end_point_coord();
    //length_of_sampling = 8.0*(0.37*this->start_point[0]/pow(navier_stokes_real.reynolds_number_inf,1.0/5.0));
    //std::cout << "Length_of_sampling for the boundary layer extraction is " << length_of_sampling << std::endl;

    //this->end_point = this->start_point+this->start_point_normal_vector*length_of_sampling;

    //coord_of_sampling.resize(this->number_of_sampling);
    //soln_of_sampling.resize(this->number_of_sampling);
    //soln_grad_of_sampling.resize(this->number_of_sampling);

    //this->evaluate_straight_line_sampling_point_soln(coord_of_sampling,soln_of_sampling,soln_grad_of_sampling);

    //this->evaluate_start_end_point_soln();

    //this->evaluate_converged_free_stream_value();

    //std::cout << "Captured U_inf is " << this->U_inf << std::endl;
    //std::cout << "Captured density_inf is " << this->density_inf << std::endl;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_functional(const bool compute_dIdW, 
                      const bool compute_dIdX, 
                      const bool compute_d2I)
{
    // Todo: this function is no longer needed once a new class is built for extraction
    (void) compute_dIdW;
    (void) compute_dIdX;
    (void) compute_d2I;

    double value = 0.0;
    return value;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
void ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_extraction_start_point_coord()
{
    if(dim==2)
        std::cout << "The specified start_point of boundary layer extraction is " << this->start_point[0] << "," << this->start_point[1] << std::endl;
    else if(dim==3)
        std::cout << "The specified start_point of boundary layer extraction is " << this->start_point[0] << "," << this->start_point[1] << "," << this->start_point[2] << std::endl;
    else
        std::cout << "The boundary layer extraction only support 2D or 3D calculations..." << std::endl;

    for (auto cell = this->dg->triangulation->begin_active(); cell != this->dg->triangulation->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                // Todo: may have a better way to guarantee the extraction point is on the boundary surface
                if (std::abs(cell->face(face)->center()[0]-this->start_point[0])<=1e-3 && cell->face(face)->center()[1]>=0.0){
                    this->start_point = cell->face(face)->center();
                }
            }
        }
    }
    if(dim==2)
        std::cout << "The (corrected) start_point of boundary layer extraction is " << this->start_point[0] << "," << this->start_point[1] << std::endl;
    else if(dim==3)
        std::cout << "The (corrected) start_point of boundary layer extraction is " << this->start_point[0] << "," << this->start_point[1] << "," << this->start_point[2] << std::endl;
    else
        std::cout << "The boundary layer extraction only support 2D or 3D calculations..." << std::endl;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
void ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_extraction_start_point_normal_tangential_vector()
{
    const auto extraction_cell = dealii::GridTools::find_active_cell_around_point(*(this->dg->triangulation),this->start_point);
    if(extraction_cell->at_boundary()){
        std::cout << "Captured cell that extraction point belongs to..." << std::endl;
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (extraction_cell->face(face)->at_boundary()) {
                this->start_point_normal_vector = extraction_cell->face(face)->get_manifold().normal_vector(extraction_cell->face(face),this->start_point);
                this->start_point_normal_vector*= -1.0;
                dealii::Point<dim,real> start_point_neighbor = this->start_point;
                // Todo: may have a better way to build a tangential vector respect to the surface geometry
                start_point_neighbor[0] += 1e-3;
                this->start_point_tangential_vector = extraction_cell->face(face)->get_manifold().get_tangent_vector(this->start_point,start_point_neighbor);
                for(int d=0;d<dim;++d){
                    this->start_point_tangential_vector[d] /= this->start_point_tangential_vector.norm();
                }
                std::cout << "Captured normal and tangential vector at extraction location..." << std::endl;
            }
        }
    } else {
        std::cout << "ERROR: Fail to capture cell that extraction point belongs to..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
void ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_extraction_end_point_coord()
{
    this->length_of_sampling = 8.0*(0.37*this->start_point[0]/pow(navier_stokes_fad.reynolds_number_inf,1.0/5.0));
    std::cout << "Length_of_sampling for the boundary layer extraction is " << this->length_of_sampling << std::endl;
    this->end_point = this->start_point+this->start_point_normal_vector*length_of_sampling;
    if(dim==2)
        std::cout << "The end_point of boundary layer extraction is " << this->end_point[0] << "," << this->end_point[1] << std::endl;
    else if(dim==3)
        std::cout << "The end_point of boundary layer extraction is " << this->end_point[0] << "," << this->end_point[1] << "," << this->end_point[2] << std::endl;
    else
        std::cout << "The boundary layer extraction only support 2D or 3D calculations..." << std::endl;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::vector<dealii::Point<dim,real>> ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_sampling_point_coord() const
{
    std::vector<dealii::Point<dim,real>> coord_of_sampling;
    coord_of_sampling.resize(number_of_sampling);
    const real dh = 1.0/number_of_sampling;
    for(int i=0;i<number_of_sampling;++i){
        const real h = (0.5+i)*dh;
        dealii::Point<dim,real> quad_coord;
        if(dim==2){
            quad_coord[0] = (1.0-h)*this->start_point[0]+h*this->end_point[0];
            quad_coord[1] = (1.0-h)*this->start_point[1]+h*this->end_point[1];
        }else if(dim==3){
            quad_coord[0] = (1.0-h)*this->start_point[0]+h*this->end_point[0];
            quad_coord[1] = (1.0-h)*this->start_point[1]+h*this->end_point[1];
            quad_coord[2] = (1.0-h)*this->start_point[2]+h*this->end_point[2];
        }
        coord_of_sampling[i] = quad_coord;
    }
    return coord_of_sampling;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::vector<dealii::Point<dim,real>> ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_total_sampling_point_coord() const
{
    std::vector<dealii::Point<dim,real>> coord_of_sampling;

    coord_of_sampling = this->evaluate_straight_line_sampling_point_coord();

    std::vector<dealii::Point<dim,real>> coord_of_total_sampling;

    coord_of_total_sampling.push_back(this->start_point);

    for(int i=0;i<number_of_sampling;++i){
        coord_of_total_sampling.push_back(coord_of_sampling[i]);
    }

    coord_of_total_sampling.push_back(this->end_point);

    return coord_of_total_sampling;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
std::vector<std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,typename dealii::Point<dim,real>>> ExtractionFunctional<dim,nstate,real,MeshType>
::find_active_cell_around_points(
        const dealii::hp::MappingCollection<dim> &mapping_collection,
        const dealii::DoFHandler<dim> &dof_handler,
        const std::vector<dealii::Point<dim,real>> &coord_of_total_sampling) const 
{
    std::vector<std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,typename dealii::Point<dim,real>>> cell_index_and_ref_points_of_total_sampling(number_of_total_sampling);
    //number_of_total_sampling = coord_of_total_sampling.size();
    //if(number_of_total_sampling != number_of_sampling+2){
    //    std::cout << "ERROR: The number of total sampling is not provided correctly..." << std::endl;
    //    std::abort();
    //}

    std::cout << "pin 0 in find_active_cell_around_points..." << std::endl;
    for(int i=0;i<number_of_total_sampling;++i){
        cell_index_and_ref_points_of_total_sampling[i] = dealii::GridTools::find_active_cell_around_point(mapping_collection,
                                                                                                          dof_handler,
                                                                                                          coord_of_total_sampling[i]);
    }

    return cell_index_and_ref_points_of_total_sampling;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
std::array<real2,nstate> ExtractionFunctional<dim,nstate,real,MeshType>
::point_value(const dealii::Point<dim,real> &coord_of_sampling,
              const dealii::hp::MappingCollection<dim> &mapping_collection,
              const dealii::hp::FECollection<dim> &fe_collection,
              const std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,typename dealii::Point<dim,real>> &cell_index_and_ref_point_of_sampling,
              const std::vector<real2> &soln_coeff,
              const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const
{
    (void) coord_of_sampling;
    const dealii::Quadrature<dim> quadrature(dealii::GeometryInfo<dim>::project_to_unit_cell(cell_index_and_ref_point_of_sampling.second));
    dealii::hp::FEValues<dim, dim> hp_fe_values(mapping_collection,
                                                fe_collection,
                                                dealii::hp::QCollection<dim>(quadrature),
                                                dealii::update_values);
    hp_fe_values.reinit(cell_index_and_ref_point_of_sampling.first);
    const dealii::FEValues<dim, dim> &fe_values = hp_fe_values.get_present_fe_values();

    //test 1
    //dealii::Vector<real> soln_coeff_vec(soln_coeff.size());
    //if constexpr(std::is_same<real2,FadFadType>::value){
    //    for(long unsigned int i=0;i<soln_coeff.size();++i){
    //        soln_coeff_vec[i] = soln_coeff[i].val().val();
    //    }
    //}
    //std::vector<dealii::Vector<real>> u_value(1, dealii::Vector<real>(fe_collection.n_components()));
    //fe_values.get_function_values(soln_coeff_vec, cell_soln_dofs_indices, u_value);
    //test 1

    //test 2
    (void) cell_soln_dofs_indices;
    std::array<real2,nstate> interpolated_soln;
    for(int i=0;i<nstate;++i){
        interpolated_soln[i] = 0.0;
    }
    const dealii::FiniteElement<dim,dim> & fe = fe_values.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_components = fe.n_components();
    for (unsigned int shape_func = 0; shape_func < dofs_per_cell;++shape_func){
        const real2 value = soln_coeff[shape_func];
        if (fe.is_primitive(shape_func)){
            const unsigned int comp = fe.system_to_component_index(shape_func).first;
            interpolated_soln[comp] += value*fe_values.shape_value(shape_func,0);
        }
        else{
            for (unsigned int c = 0; c < n_components; ++c){
                interpolated_soln[c] += value*fe_values.shape_value_component(shape_func,0,c);
            }
        }
    }
    //test 2


    //std::vector<dealii::Vector<real2>> u_value(1, dealii::Vector<real2>(fe_collection.n_components()));
    //fe_values.get_function_values(soln_coeff, cell_soln_dofs_indices, u_value);

    //std::array<real2,nstate> interpolated_soln;
    //for(int i=0;i<nstate;++i){
    //    interpolated_soln[i] = u_value[0][i];
    //}

    return interpolated_soln;
}

//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> ExtractionFunctional<dim,nstate,real,MeshType>
::point_gradient(const dealii::Point<dim,real> &coord_of_sampling,
                 const dealii::hp::MappingCollection<dim> &mapping_collection,
                 const dealii::hp::FECollection<dim> &fe_collection,
                 const std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,typename dealii::Point<dim,real>> &cell_index_and_ref_point_of_sampling,
                 const std::vector<real2> &soln_coeff,
                 const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const
{
    (void) coord_of_sampling;
    const dealii::Quadrature<dim> quadrature(dealii::GeometryInfo<dim>::project_to_unit_cell(cell_index_and_ref_point_of_sampling.second));
    dealii::hp::FEValues<dim, dim> hp_fe_values(mapping_collection,
                                                fe_collection,
                                                dealii::hp::QCollection<dim>(quadrature),
                                                dealii::update_gradients);
    hp_fe_values.reinit(cell_index_and_ref_point_of_sampling.first);
    const dealii::FEValues<dim, dim> &fe_values = hp_fe_values.get_present_fe_values();

    //test 1
    //dealii::Vector<real> soln_coeff_vec(soln_coeff.size());
    //if constexpr(std::is_same<real2,FadFadType>::value){
    //    for(long unsigned int i=0;i<soln_coeff.size();++i){
    //        soln_coeff_vec[i] = soln_coeff[i].val().val();
    //    }
    //}
    //std::vector<std::vector<dealii::Tensor<1, dim, real>>> u_gradient(1, std::vector<dealii::Tensor<1, dim, real>>(fe_collection.n_components()));
    //fe_values.get_function_gradients(soln_coeff_vec, cell_soln_dofs_indices, u_gradient);
    //test 1

    //test 2
    (void) cell_soln_dofs_indices;
    std::array<dealii::Tensor<1,dim,real2>,nstate> interpolated_soln_grad;
    for(int i=0;i<nstate;++i){
        for(int j=0;j<dim;++j){
            interpolated_soln_grad[i][j] = 0.0;
        }
    }
    const dealii::FiniteElement<dim,dim> & fe = fe_values.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_components = fe.n_components();
    for (unsigned int shape_func = 0; shape_func < dofs_per_cell;++shape_func){
        const real2 value = soln_coeff[shape_func];
        if (fe.is_primitive(shape_func)){
            const unsigned int comp = fe.system_to_component_index(shape_func).first;
            interpolated_soln_grad[comp] += value*fe_values.shape_grad(shape_func,0);
        }
        else{
            for (unsigned int c = 0; c < n_components; ++c){
                interpolated_soln_grad[c] += value*fe_values.shape_grad_component(shape_func,0,c);
            }
        }
    }
    //test 2

    //std::vector<std::vector<dealii::Tensor<1, dim, real2>>> u_gradient(1, std::vector<dealii::Tensor<1, dim, real2>>(fe_collection.n_components()));
    //fe_values.get_function_gradients(soln_coeff, cell_soln_dofs_indices, u_gradient);

    //std::array<dealii::Tensor<1,dim,real2>,nstate> interpolated_soln_grad;
    //for(int i=0;i<nstate;++i){
    //    for(int j=0;j<dim;++j){
    //        interpolated_soln_grad[i][j] = u_gradient[0][i][j];
    //    }
    //}

    return interpolated_soln_grad;
}
//----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//template <typename real2>
//std::vector<std::array<real2,nstate>> ExtractionFunctional<dim,nstate,real,MeshType>
//::evaluate_straight_line_sampling_point_soln(const std::vector<dealii::Point<dim,real>> &coord_of_sampling,
//                                             const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector)
//{
//    std::vector<std::array<real2,nstate>> soln_of_sampling
//    soln_of_sampling.resize(number_of_sampling);
//    for(int i=0;i<number_of_sampling;++i){
//        dealii::Vector<real2> soln;
//        dealii::VectorTools::point_value(this->dg->dof_handler,solution_vector,coord_of_sampling[i],soln);
//        std::array<real2,nstate> soln_at_q;
//        for(int s=0;s<nstate;++s){
//            soln_at_q[s] = soln[s];
//        }
//        soln_of_sampling[i] = soln_at_q;
//    }
//    return soln_of_sampling;
//}
//----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//template <typename real2>
//std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> ExtractionFunctional<dim,nstate,real,MeshType>
//::evaluate_straight_line_sampling_point_soln_grad(const std::vector<dealii::Point<dim,real>> &coord_of_sampling,
//                                                  const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector)
//{
//    std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> soln_grad_of_sampling;
//    soln_grad_of_sampling.resize(number_of_sampling);
//    for(int i=0;i<number_of_sampling;++i){
//        std::vector<dealii::Tensor<1,dim,real2>> soln_grad;
//        dealii::VectorTools::point_gradient(this->dg->dof_handler,solution_vector,coord_of_sampling[i],soln_grad);
//        std::array<dealii::Tensor<1,dim,real2>,nstate> soln_grad_at_q;
//        for(int s=0;s<nstate;++s){
//            soln_grad_at_q[s] = soln_grad[s];
//        }
//        soln_grad_of_sampling[i] = soln_grad_at_q;
//    }
//    return soln_grad_of_sampling;
//}
//----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//template <typename real2>
//std::array<real2,nstate> ExtractionFunctional<dim,nstate,real,MeshType>
//::evaluate_start_point_soln(const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector)
//{
//    std::array<real2,nstate> soln_at_start_point;
//    dealii::Vector<real2> soln;
//    dealii::VectorTools::point_value(this->dg->dof_handler,solution_vector,this->start_point,soln);
//    for(int s=0;s<nstate;++s){
//        soln_at_start_point[s] = soln[s];
//    }
//    return soln_at_start_point;
//}
//----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//template <typename real2>
//std::array<real2,nstate> ExtractionFunctional<dim,nstate,real,MeshType>
//::evaluate_end_point_soln(const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector)
//{
//    std::array<real2,nstate> soln_at_end_point;
//    dealii::Vector<real2> soln;
//    dealii::VectorTools::point_value(this->dg->dof_handler,solution_vector,this->end_point,soln);
//    for(int s=0;s<nstate;++s){
//        soln_at_end_point[s] = soln[s];
//    }
//    return soln_at_end_point;
//}
//----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//template <typename real2>
//std::array<dealii::Tensor<1,dim,real2>,nstate> ExtractionFunctional<dim,nstate,real,MeshType>
//::evaluate_start_point_soln_grad(const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector)
//{
//    std::array<dealii::Tensor<1,dim,real2>,nstate> soln_grad_at_start_point;
//    std::vector<dealii::Tensor<1,dim,real2>> soln_grad;
//    dealii::VectorTools::point_gradient(this->dg->dof_handler,solution_vector,this->start_point,soln_grad);
//    for(int s=0;s<nstate;++s){
//        soln_grad_at_start_point[s] = soln_grad[s];
//    }
//    return soln_grad_at_start_point;
//}
//----------------------------------------------------------------
//template <int dim,int nstate,typename real,typename MeshType>
//template <typename real2>
//std::array<dealii::Tensor<1,dim,real2>,nstate> ExtractionFunctional<dim,nstate,real,MeshType>
//::evaluate_end_point_soln_grad(const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector)
//{
//    std::array<dealii::Tensor<1,dim,real2>,nstate> soln_grad_at_end_point;
//    std::vector<dealii::Tensor<1,dim,real>> soln_grad;
//    dealii::VectorTools::point_gradient(this->dg->dof_handler,solution_vector,this->end_point,soln_grad);
//    for(int s=0;s<nstate;++s){
//        soln_grad_at_end_point[s] = soln_grad[s];
//    }
//    return soln_grad_at_end_point;
//}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_integral(const Functional_types &functional_type,
                                  const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    const std::pair<real,real> values_free_stream = evaluate_converged_free_stream_values(soln_of_total_sampling);
    const real dh = 1.0/number_of_sampling;
    const real integral_variable = this->length_of_sampling*dh;
    real2 integral = 0.0;
    for(int i=1;i<=number_of_sampling;++i){
        real2 integrand = this->evaluate_straight_line_integrand(functional_type,values_free_stream,this->start_point_tangential_vector,soln_of_total_sampling[i]);
        integral += integrand*integral_variable;
    }

    return integral;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_integrand(const Functional_types &functional_type,
                                   const std::pair<real,real> &values_free_stream,
                                   const dealii::Tensor<1,dim,real> &tangential,
                                   const std::array<real2,nstate> &soln_at_q) const
{
    switch(functional_type) {
        case Functional_types::displacement_thickness : 
            return this->evaluate_displacement_thickness_integrand(values_free_stream,
                                                                   tangential,
                                                                   soln_at_q);
            break;
        case Functional_types::momentum_thickness : 
            return this->evaluate_momentum_thickness_integrand(values_free_stream,
                                                               tangential,
                                                               soln_at_q);
            break;
        default: 
            break;
    }
    std::cout << "ERROR: Fail to determine integrand for line integral..." << std::endl;
    std::abort();

}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_displacement_thickness_integrand(const std::pair<real,real> &values_free_stream,
                                            const dealii::Tensor<1,dim,real> &tangential,
                                            const std::array<real2,nstate> &soln_at_q) const
{
    std::array<real2,dim+2> ns_soln_at_q;
    if constexpr(nstate==dim+2){
        ns_soln_at_q = soln_at_q;
    } else if constexpr(nstate==dim+3){
        ns_soln_at_q = rans_sa_neg_fad->extract_rans_conservative_solution(soln_at_q);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    const real2 density_at_q = ns_soln_at_q[0];
    const dealii::Tensor<1,dim,real2> velocity_at_q = navier_stokes_fad.compute_velocities(ns_soln_at_q);
    real2 U_tangential_at_q = 0.0;
    for(int d=0;d<dim;++d){
        U_tangential_at_q += velocity_at_q[d]*tangential[d];
    }

    real speed_free_stream = values_free_stream.first;
    real density_free_stream = values_free_stream.second;
    return 1.0-density_at_q*U_tangential_at_q/(density_free_stream*speed_free_stream);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_momentum_thickness_integrand(const std::pair<real,real> &values_free_stream,
                                        const dealii::Tensor<1,dim,real> &tangential,
                                        const std::array<real2,nstate> &soln_at_q) const
{
    std::array<real2,dim+2> ns_soln_at_q;
    if constexpr(nstate==dim+2){
        ns_soln_at_q = soln_at_q;
    } else if constexpr(nstate==dim+3){
        ns_soln_at_q = rans_sa_neg_fad->extract_rans_conservative_solution(soln_at_q);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    const real2 density_at_q = ns_soln_at_q[0];
    const dealii::Tensor<1,dim,real2> velocity_at_q = navier_stokes_fad.compute_velocities(ns_soln_at_q);
    real2 U_tangential_at_q = 0.0;
    for(int d=0;d<dim;++d){
        U_tangential_at_q += velocity_at_q[d]*tangential[d];
    }

    real speed_free_stream = values_free_stream.first;
    real density_free_stream = values_free_stream.second;
    return density_at_q*U_tangential_at_q/(density_free_stream*speed_free_stream)*(1.0-U_tangential_at_q/speed_free_stream);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_kinematic_viscosity(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    real2 density_at_wall;
    real2 dynamic_viscosity_at_wall;
    real2 kinematic_viscosity_at_wall;
    std::array<real2,dim+2> ns_soln_of_sampling;
    if constexpr(nstate==dim+2){
        ns_soln_of_sampling = soln_of_total_sampling[0];
    } else if constexpr(nstate==dim+3){
        ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_total_sampling[0]);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }

    const std::array<real2,dim+2> primitive_soln_at_wall = navier_stokes_fad.convert_conservative_to_primitive(ns_soln_of_sampling);
    density_at_wall = primitive_soln_at_wall[0];
    dynamic_viscosity_at_wall = navier_stokes_fad.compute_viscosity_coefficient(primitive_soln_at_wall);
    kinematic_viscosity_at_wall = dynamic_viscosity_at_wall/density_at_wall;

    return kinematic_viscosity_at_wall;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_displacement_thickness(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    const Functional_types functional_type = Functional_types::displacement_thickness;

    return this->evaluate_straight_line_integral(functional_type,soln_of_total_sampling);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_momentum_thickness(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    const Functional_types functional_type = Functional_types::momentum_thickness;

    return this->evaluate_straight_line_integral(functional_type,soln_of_total_sampling);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_edge_velocity(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    //const real tolerance = 1e-3;
    //real2 edge_velocity;
    //for(int i=0;i<number_of_sampling;++i){
    //    std::array<real2,dim+2> ns_soln_of_sampling;
    //    if constexpr(nstate==dim+2){
    //        ns_soln_of_sampling = soln_of_sampling[i];
    //    } else if constexpr(nstate==dim+3){
    //        ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_sampling[i]);
    //    } else {
    //        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
    //        std::abort();
    //    }
    //    edge_velocity = navier_stokes_fad.compute_velocities(ns_soln_of_sampling).norm();
    //    if ((edge_velocity-0.99*U_inf)<=tolerance){
    //        std::cout << "Captured edge vbelocity..." << std::endl;
    //        return edge_velocity;
    //    }
    //}
    //std::cout << "ERROR: Fail to capture edge velocity..." << std::endl;
    //std::abort();

    const std::pair<real,real> values_free_stream = evaluate_converged_free_stream_values(soln_of_total_sampling);
    return 0.99*values_free_stream.first;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_shear_stress(const std::array<real2,nstate> &soln_at_q,
                        const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q) const
{
    std::array<real2,dim+2> ns_soln_at_q;
    std::array<dealii::Tensor<1,dim,real2>,dim+2> ns_soln_grad_at_q;
    if constexpr(nstate==dim+2){
        ns_soln_at_q = soln_at_q;
        ns_soln_grad_at_q = soln_grad_at_q;
    } else if constexpr(nstate==dim+3){
        ns_soln_at_q = rans_sa_neg_fad->extract_rans_conservative_solution(soln_at_q);
        ns_soln_grad_at_q = rans_sa_neg_fad->extract_rans_solution_gradient(soln_grad_at_q);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    real2 shear_stress_at_q;
    const std::array<real2,dim+2> primitive_soln_at_q = navier_stokes_fad.convert_conservative_to_primitive(ns_soln_at_q);
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> primitive_soln_grad_at_q = navier_stokes_fad.convert_conservative_gradient_to_primitive_gradient(ns_soln_at_q,ns_soln_grad_at_q);
    dealii::Tensor<2,dim,real2> viscous_stress_tensor_at_q = navier_stokes_fad.compute_viscous_stress_tensor(primitive_soln_at_q,primitive_soln_grad_at_q);
    dealii::Tensor<1,dim,real2> shear_stress_vector_at_q;
    for (int r=0;r<dim;++r){
        for (int c=0;c<dim;++c){
            shear_stress_vector_at_q[r] = viscous_stress_tensor_at_q[r][c]*this->start_point_normal_vector[c];
        }
    }
    shear_stress_at_q = shear_stress_vector_at_q.norm();
    return shear_stress_at_q;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_wall_shear_stress(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
                             const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const
{
    return evaluate_shear_stress(soln_of_total_sampling[0],soln_grad_of_total_sampling[0]);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_maximum_shear_stress(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
                                const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const
{
    real2 maximum_shear_stress = evaluate_wall_shear_stress(soln_of_total_sampling,soln_grad_of_total_sampling);
    for(int i=1;i<number_of_total_sampling;++i){
        real2 maximum_shear_stress_n;
        maximum_shear_stress_n = evaluate_shear_stress(soln_of_total_sampling[i],soln_grad_of_total_sampling[i]);
        maximum_shear_stress = maximum_shear_stress > maximum_shear_stress_n ? maximum_shear_stress : maximum_shear_stress_n;
    }
    if constexpr(std::is_same<real2,real>::value){
        return maximum_shear_stress;
    } 
    else if constexpr(std::is_same<real2,FadType>::value){
        return maximum_shear_stress.val();
    }
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_friction_velocity(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
                             const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const
{
    real2 wall_shear_stress = this->evaluate_wall_shear_stress(soln_of_total_sampling,soln_grad_of_total_sampling);
    std::array<real2,nstate> soln_at_wall = soln_of_total_sampling[0];
    real2 density_at_wall = soln_at_wall[0];
    return sqrt(wall_shear_stress/density_at_wall);
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_boundary_layer_thickness(const std::vector<dealii::Point<dim,real>> &coord_of_total_sampling,
                                    const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    const std::pair<real,real> values_free_stream = evaluate_converged_free_stream_values(soln_of_total_sampling);
    const real tolerance = 1e-3;
    real boundary_layer_thickness;
    std::array<real2,dim+2> ns_soln_of_sampling;
    if constexpr(nstate==dim+2){
        ns_soln_of_sampling = soln_of_total_sampling[0];
    } else if constexpr(nstate==dim+3){
        ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_total_sampling[0]);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    real2 edge_velocity = navier_stokes_fad.compute_velocities(ns_soln_of_sampling).norm();
    for(int i=1;i<number_of_total_sampling;++i){
        if constexpr(nstate==dim+2){
            ns_soln_of_sampling = soln_of_total_sampling[i];
        } else if constexpr(nstate==dim+3){
            ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_total_sampling[i]);
        } else {
            std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
            std::abort();
        }
        real2 edge_velocity_n = navier_stokes_fad.compute_velocities(ns_soln_of_sampling).norm();
        if(edge_velocity_n>=0.99*values_free_stream.first && std::abs(edge_velocity-edge_velocity_n)<=tolerance) {
            boundary_layer_thickness = this->start_point.distance(coord_of_total_sampling[i]);
            std::cout << "Captured boundary layer thickness..." << std::endl;
            return boundary_layer_thickness;
        } else {
            edge_velocity = edge_velocity_n;
        }
    }
    std::cout << "ERROR: Fail to capture boundary layer thickness..." << std::endl;
    std::abort();
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
real2 ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_pressure_gradient_tangential(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
                                        const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const
{
    std::array<real2,dim+2> ns_soln_of_sampling;
    std::array<dealii::Tensor<1,dim,real2>,dim+2> ns_soln_grad_of_sampling;
    if constexpr(nstate==dim+2){
        ns_soln_of_sampling = soln_of_total_sampling[0];
        ns_soln_grad_of_sampling = soln_grad_of_total_sampling[0];
    } else if constexpr(nstate==dim+3){
        ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_total_sampling[0]);
        ns_soln_grad_of_sampling = rans_sa_neg_fad->extract_rans_solution_gradient(soln_grad_of_total_sampling[0]);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }

    dealii::Tensor<1,dim,real2> pressure_gradient = navier_stokes_fad.convert_conservative_gradient_to_primitive_gradient(ns_soln_of_sampling,ns_soln_grad_of_sampling)[dim+2-1];
    real2 pressure_gradient_tangential = 0.0;
    for(int d=0;d<dim;++d){
        pressure_gradient_tangential += pressure_gradient[d]*this->start_point_tangential_vector[d];
    }
    return pressure_gradient_tangential;
}
//----------------------------------------------------------------
template <int dim,int nstate,typename real,typename MeshType>
template <typename real2>
std::pair<real,real> ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_converged_free_stream_values(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const
{
    std::pair<real,real> values_free_stream;
    real speed_free_stream;
    real density_free_stream;
    const real2 tolerance = 1e-4;
    int consecutive_counter = 0;
    bool consecutive_sensor = false;
    std::array<real2,dim+2> ns_soln_of_sampling;
    if constexpr(nstate==dim+2){
        ns_soln_of_sampling = soln_of_total_sampling[0];
    } else if constexpr(nstate==dim+3){
        ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_total_sampling[0]);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    real2 speed_sampling = navier_stokes_fad.compute_velocities(ns_soln_of_sampling).norm();
    real2 speed_sampling_n;
    for(int i=1;i<number_of_total_sampling;++i){
        if constexpr(nstate==dim+2){
            ns_soln_of_sampling = soln_of_total_sampling[i];
        } else if constexpr(nstate==dim+3){
            ns_soln_of_sampling = rans_sa_neg_fad->extract_rans_conservative_solution(soln_of_total_sampling[i]);
        } else {
            std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
            std::abort();
        }
        speed_sampling_n = navier_stokes_fad.compute_velocities(ns_soln_of_sampling).norm();
        if(std::abs(speed_sampling-speed_sampling_n)<=tolerance){
            if(consecutive_sensor){
                if constexpr(std::is_same<real2,real>::value){
                    speed_free_stream = speed_sampling_n;
                    density_free_stream = ns_soln_of_sampling[0];
                } 
                else if constexpr(std::is_same<real2,FadType>::value){
                    speed_free_stream = speed_sampling_n.val();
                    density_free_stream = ns_soln_of_sampling[0].val();
                }
                values_free_stream.first = speed_free_stream;
                values_free_stream.second = density_free_stream;
                std::cout << "Captured converged free stream values..." << std::endl;
                return values_free_stream;
            }
            consecutive_counter++;
            if(consecutive_counter>=10)
                consecutive_sensor = true;
        } else {
            speed_sampling = speed_sampling_n;
        }
    }
    std::cout << "ERROR: Fail to capture converged free stream values..." << std::endl;
    std::abort();
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- ExtractionFunctional
#if PHILIP_DIM!=1
template class ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template std::array<FadFadType,PHILIP_DIM+2> 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::point_value(
    const dealii::Point<PHILIP_DIM,double> &coord_of_sampling,
    const dealii::hp::MappingCollection<PHILIP_DIM> &mapping_collection,
    const dealii::hp::FECollection<PHILIP_DIM> &fe_collection,
    const std::pair<typename dealii::DoFHandler<PHILIP_DIM>::active_cell_iterator,typename dealii::Point<PHILIP_DIM,double>> &cell_index_and_ref_point_of_sampling,
    const std::vector<FadFadType> &soln_coeff,
    const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const;
template std::array<FadFadType,PHILIP_DIM+3> 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::point_value(
    const dealii::Point<PHILIP_DIM,double> &coord_of_sampling,
    const dealii::hp::MappingCollection<PHILIP_DIM> &mapping_collection,
    const dealii::hp::FECollection<PHILIP_DIM> &fe_collection,
    const std::pair<typename dealii::DoFHandler<PHILIP_DIM>::active_cell_iterator,typename dealii::Point<PHILIP_DIM,double>> &cell_index_and_ref_point_of_sampling,
    const std::vector<FadFadType> &soln_coeff,
    const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const;

template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::point_gradient(
    const dealii::Point<PHILIP_DIM,double> &coord_of_sampling,
    const dealii::hp::MappingCollection<PHILIP_DIM> &mapping_collection,
    const dealii::hp::FECollection<PHILIP_DIM> &fe_collection,
    const std::pair<typename dealii::DoFHandler<PHILIP_DIM>::active_cell_iterator,typename dealii::Point<PHILIP_DIM,double>> &cell_index_and_ref_point_of_sampling,
    const std::vector<FadFadType> &soln_coeff,
    const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+3> 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::point_gradient(
    const dealii::Point<PHILIP_DIM,double> &coord_of_sampling,
    const dealii::hp::MappingCollection<PHILIP_DIM> &mapping_collection,
    const dealii::hp::FECollection<PHILIP_DIM> &fe_collection,
    const std::pair<typename dealii::DoFHandler<PHILIP_DIM>::active_cell_iterator,typename dealii::Point<PHILIP_DIM,double>> &cell_index_and_ref_point_of_sampling,
    const std::vector<FadFadType> &soln_coeff,
    const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const;

template std::pair<double,double> 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_converged_free_stream_values(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling) const;
template std::pair<double,double> 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_converged_free_stream_values(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling) const;

template double 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_boundary_layer_thickness(
    const std::vector<dealii::Point<PHILIP_DIM,double>> &coord_of_total_sampling,
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling) const;
template double 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_boundary_layer_thickness(
    const std::vector<dealii::Point<PHILIP_DIM,double>> &coord_of_total_sampling,
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling) const;

template double 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_edge_velocity(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling) const;
template double 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_edge_velocity(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling) const;

template double 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_maximum_shear_stress(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2>> &soln_grad_of_total_sampling) const;
template double 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_maximum_shear_stress(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3>> &soln_grad_of_total_sampling) const;

template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_displacement_thickness(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling) const;
template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_displacement_thickness(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling) const;

template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_momentum_thickness(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling) const;
template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_momentum_thickness(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling) const;

template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_friction_velocity(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2>> &soln_grad_of_total_sampling) const;
template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_friction_velocity(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3>> &soln_grad_of_total_sampling) const;

template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_pressure_gradient_tangential(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2>> &soln_grad_of_total_sampling) const;
template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_pressure_gradient_tangential(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3>> &soln_grad_of_total_sampling) const;

template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_wall_shear_stress(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2>> &soln_grad_of_total_sampling) const;
template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_wall_shear_stress(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling,
    const std::vector<std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3>> &soln_grad_of_total_sampling) const;

template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_kinematic_viscosity(
    const std::vector<std::array<FadType,PHILIP_DIM+2>> &soln_of_total_sampling) const;
template FadType 
ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::evaluate_kinematic_viscosity(
    const std::vector<std::array<FadType,PHILIP_DIM+3>> &soln_of_total_sampling) const;
#endif
} // PHiLiP namespace