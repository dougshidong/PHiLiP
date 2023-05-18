#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "physics/physics_factory.h"
#include "physics/model_factory.h"

#include "extraction_functional.hpp"

#include <memory>

namespace PHiLiP {

template <int dim,int nstate,typename real,typename MeshType>
ExtractionFunctional<dim,nstate,real,MeshType>
::ExtractionFunctional(
    std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
    const dealii::Point<dim,real> start_point_input,
    const int number_of_sampling_input)
    : Functional<dim,nstate,real,MeshType>(dg_input)
    , start_point(start_point_input)
    , number_of_sampling(number_of_sampling_input)
    //, navier_stokes_real(dynamic_cast< Physics::NavierStokes<dim,dim+2,real> &>(*(this->physics_real)))
    , navier_stokes_real(dynamic_cast< Physics::NavierStokes<dim,dim+2,real> &>(*(PHiLiP::Physics::PhysicsFactory<dim,dim+2,real>::create_Physics(this->dg->all_parameters, Parameters::AllParameters::PartialDifferentialEquation::navier_stokes))))
{
    if (nstate==dim+3){
        rans_sa_neg_real = std::dynamic_pointer_cast< Physics::ReynoldsAveragedNavierStokes_SAneg<dim,dim+3,real> >(PHiLiP::Physics::ModelFactory<dim,dim+3,real>::create_Model(this->dg->all_parameters)); 

    }

//    const auto extraction_cell = dealii::GridTools::find_active_cell_around_point(*(this->dg->triangulation),start_point);
//    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
//        if (extraction_cell->face(face)->at_boundary()) {
//            //const auto extraction_manifold = extraction_cell->face(face)->get_manifold();
//
//            //const dealii::types::manifold_id manifold_id_on_boundary = 0;
//            //this->dg->triangulation->set_all_manifold_ids_on_boundary(extraction_cell->face(face)->boundary_id (),manifold_id_on_boundary);
//            //const auto extraction_manifold = this->dg->triangulation->get_manifold(manifold_id_on_boundary);
//
//            //start_point_normal_vector = extraction_manifold.normal_vector(extraction_cell->face(face),start_point);
//            start_point_normal_vector = extraction_cell->face(face)->get_manifold().normal_vector(extraction_cell->face(face),start_point);
//            // ToDo: check the right function for tangential vector
//            //start_point_tangential_vector = extraction_manifold.normal_vector(extraction_cell->face(face),start_point);
//            dealii::Point<dim,real> start_point_neighbor = start_point;
//            start_point_neighbor[0] += 1e-3;
//            start_point_tangential_vector = extraction_cell->face(face)->get_manifold().get_tangent_vector(start_point,start_point_neighbor);
//        } else {
//            std::cout << "ERROR: Fail to load the normal vector for the extraction start point..." << std::endl;
//            std::abort();
//        }
//    }
    start_point_normal_vector[0] = 0.0;
    start_point_normal_vector[1] = 1.0;

    start_point_tangential_vector[0] = 1.0;
    start_point_tangential_vector[0] = 0.0;

    // ToDo: check dimensionalization of length_of_sampling (should be fine)
    length_of_sampling = 4.0*(0.37*start_point[0]/pow(navier_stokes_real.density_inf*navier_stokes_real.velocities_inf.norm()*start_point[0]/navier_stokes_real.viscosity_coefficient_inf,1.0/5.0));
    std::cout << "length_of_sampling is " << length_of_sampling << std::endl;
    length_of_sampling = 0.3;
    // ToDo: check the operator (should be fine)
    end_point = start_point+start_point_normal_vector*length_of_sampling;

    coord_of_sampling.resize(number_of_sampling);
    soln_of_sampling.resize(number_of_sampling);
    soln_grad_of_sampling.resize(number_of_sampling);

    this->evaluate_straight_line_sampling_point_soln();

    this->evaluate_start_end_point_soln();

    this->evaluate_converged_free_stream_value();
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_functional(const bool compute_dIdW, 
                      const bool compute_dIdX, 
                      const bool compute_d2I)
{
    double value = Functional<dim,nstate,real,MeshType>::evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);

    return value;
}

template <int dim,int nstate,typename real,typename MeshType>
void ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_sampling_point_soln()
{
    //const real dt = sqrt((end_point[0]-start_point[0])*(end_point[0]-start_point[0])
    //                    +(end_point[1]-start_point[1])*(end_point[1]-start_point[1]))/number_of_sampling;
    const real dt = start_point.distance(end_point)/number_of_sampling;
    for(int i=0;i<number_of_sampling;++i){

        const real t = (0.5+i)*dt;
        dealii::Point<dim,real> quad_coord;
        quad_coord[0] = (1.0-t)*start_point[0]+t*end_point[0];
        quad_coord[1] = (1.0-t)*start_point[1]+t*end_point[1];
        this->coord_of_sampling[i] = quad_coord;

        dealii::Vector<real> soln;
        dealii::VectorTools::point_value(this->dg->dof_handler,this->dg->solution,quad_coord,soln);
        std::array<real,nstate> soln_at_q;
        for(int s=0;s<nstate;++s){
            soln_at_q[s] = soln[s];
        }
        this->soln_of_sampling[i] = soln_at_q;

        std::vector<dealii::Tensor<1,dim,real>> soln_grad;
        dealii::VectorTools::point_gradient(this->dg->dof_handler,this->dg->solution,quad_coord,soln_grad);
        std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_q;
        for(int s=0;s<nstate;++s){
            soln_grad_at_q[s] = soln_grad[s];
        }
        this->soln_grad_of_sampling[i] = soln_grad_at_q;
    }
}

template <int dim,int nstate,typename real,typename MeshType>
void ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_start_end_point_soln()
{
    dealii::Vector<real> soln;
    dealii::VectorTools::point_value(this->dg->dof_handler,this->dg->solution,start_point,soln);
    for(int s=0;s<nstate;++s){
        soln_at_start_point[s] = soln[s];
    }
    dealii::VectorTools::point_value(this->dg->dof_handler,this->dg->solution,end_point,soln);
    for(int s=0;s<nstate;++s){
        soln_at_end_point[s] = soln[s];
    }

    std::vector<dealii::Tensor<1,dim,real>> soln_grad;
    dealii::VectorTools::point_gradient(this->dg->dof_handler,this->dg->solution,start_point,soln_grad);
    for(int s=0;s<nstate;++s){
        soln_grad_at_start_point[s] = soln_grad[s];
    }
    dealii::VectorTools::point_gradient(this->dg->dof_handler,this->dg->solution,end_point,soln_grad);
    for(int s=0;s<nstate;++s){
        soln_grad_at_end_point[s] = soln_grad[s];
    }
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_integral(const dealii::Point<dim,real> &start_point,
                                  const dealii::Point<dim,real> &end_point) const
{
    //const real parameterized_factor = sqrt((end_point[0]-start_point[0])*(end_point[0]-start_point[0])
    //                                      +(end_point[1]-start_point[1])*(end_point[1]-start_point[1]));
    //const real dt = sqrt((end_point[0]-start_point[0])*(end_pointd[0]-start_point[0])
    //                    +(end_point[1]-start_point[1])*(end_pointd[1]-start_point[1]))/number_of_sampling;
    const real parameterized_factor = start_point.distance(end_point);
    const real dt = parameterized_factor/number_of_sampling;
    const real integral_variable = parameterized_factor*dt;
    real integral = 0.0;
    for(int i=0;i<number_of_sampling;i++){
        real integrand = this->evaluate_straight_line_integrand(start_point_tangential_vector,soln_of_sampling[i],soln_grad_of_sampling[i]);
        integral += integrand*integral_variable;
    }

    return integral;
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_straight_line_integrand(const dealii::Tensor<1,dim,real> &normal,
                                   const std::array<real,nstate> &soln_at_q,
                                   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const
{
    switch(this->functional_type) {
        case Functional_types::displacement_thickness : 
            return this->evaluate_displacement_thickness_integrand(
                        normal,
                        soln_at_q,
                        soln_grad_at_q);
            break;
        case Functional_types::momentum_thickness : 
            return this->evaluate_momentum_thickness_integrand(
                        normal,
                        soln_at_q,
                        soln_grad_at_q);
            break;
        default: 
            break;
    }
    std::cout << "ERROR: Fail to determine integrand for line integral..." << std::endl;
    std::abort();

}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_displacement_thickness_integrand(const dealii::Tensor<1,dim,real> &normal,
                                            const std::array<real,nstate> &soln_at_q,
                                            const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_at_q*/) const
{
    std::array<real,dim+2> ns_soln_at_q;
    if constexpr(nstate==dim+2){
        ns_soln_at_q = soln_at_q;
    } else if constexpr(nstate==dim+3){
        ns_soln_at_q = rans_sa_neg_real->extract_rans_conservative_solution(soln_at_q);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    const real density = ns_soln_at_q[0];
    const dealii::Tensor<1,dim,real> velocity = navier_stokes_real.compute_velocities(ns_soln_at_q);
    real U_tangential = 0.0;
    for(int d=0;d<dim;++d){
        U_tangential += velocity[d]*normal[d];
    }

    return 1.0-density*U_tangential/(density_inf*U_inf);
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_momentum_thickness_integrand(const dealii::Tensor<1,dim,real> &normal,
                                        const std::array<real,nstate> &soln_at_q,
                                        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_at_q*/) const
{
    std::array<real,dim+2> ns_soln_at_q;
    if constexpr(nstate==dim+2){
        ns_soln_at_q = soln_at_q;
    } else if constexpr(nstate==dim+3){
        ns_soln_at_q = rans_sa_neg_real->extract_rans_conservative_solution(soln_at_q);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    const real density = ns_soln_at_q[0];
    const dealii::Tensor<1,dim,real> velocity = navier_stokes_real.compute_velocities(ns_soln_at_q);
    real U_tangential = 0.0;
    for(int d=0;d<dim;++d){
        U_tangential += velocity[d]*normal[d];
    }

    return density*U_tangential/(density_inf*U_inf)*(1.0-U_tangential/U_inf);
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_displacement_thickness()
{
    this->functional_type = Functional_types::displacement_thickness;

    return this->evaluate_straight_line_integral(this->start_point,this->end_point);
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_momentum_thickness()
{
    this->functional_type = Functional_types::momentum_thickness;

    return this->evaluate_straight_line_integral(this->start_point,this->end_point);
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_edge_velocity()
{
    const real tolerance = 1e-5;
    real edge_velocity;
    for(int i=0;i<number_of_sampling;++i){
        std::array<real,dim+2> ns_soln_of_sampling;
        if constexpr(nstate==dim+2){
            ns_soln_of_sampling = soln_of_sampling[i];
        } else if constexpr(nstate==dim+3){
            ns_soln_of_sampling = rans_sa_neg_real->extract_rans_conservative_solution(soln_of_sampling[i]);
        } else {
            std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
            std::abort();
        }
        edge_velocity = navier_stokes_real.compute_velocities(ns_soln_of_sampling).norm();
        if (std::abs(edge_velocity-0.99*U_inf)<=tolerance){
            std::cout << "Captured edge vbelocity..." << std::endl;
            return edge_velocity;
        }
    }
    std::cout << "ERROR: Fail to capture edge velocity..." << std::endl;
    std::abort();
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_wall_shear_stress(){
    std::array<real,dim+2> ns_soln_at_start_point;
    std::array<dealii::Tensor<1,dim,real>,dim+2> ns_soln_grad_at_start_point;
    if constexpr(nstate==dim+2){
        ns_soln_at_start_point = this->soln_at_start_point;
        ns_soln_grad_at_start_point = this->soln_grad_at_start_point;
    } else if constexpr(nstate==dim+3){
        ns_soln_at_start_point = rans_sa_neg_real->extract_rans_conservative_solution(this->soln_at_start_point);
        ns_soln_grad_at_start_point = rans_sa_neg_real->extract_rans_solution_gradient(this->soln_grad_at_start_point);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    real wall_shear_stress;
    const std::array<real,dim+2> primitive_soln_at_start_point = navier_stokes_real.convert_conservative_to_primitive(ns_soln_at_start_point);
    const std::array<dealii::Tensor<1,dim,real>,dim+2> primitive_soln_grad_at_start_point = navier_stokes_real.convert_conservative_gradient_to_primitive_gradient(ns_soln_at_start_point,ns_soln_grad_at_start_point);
    dealii::Tensor<2,dim,real> viscous_stress_tensor = navier_stokes_real.compute_viscous_stress_tensor(primitive_soln_at_start_point,primitive_soln_grad_at_start_point);
    dealii::Tensor<1,dim,real> wall_shear_stress_vector;
    for (int r=0;r<dim;++r){
        for (int c=0;c<dim;++c){
            wall_shear_stress_vector[r] = viscous_stress_tensor[r][c]*start_point_normal_vector[c];
        }
    }
    wall_shear_stress = std::abs(wall_shear_stress_vector.norm());
    return wall_shear_stress;
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_friction_velocity(){
    real wall_shear_stress = this->evaluate_wall_shear_stress();
    real density_at_wall = soln_at_start_point[0];
    return sqrt(wall_shear_stress/density_at_wall);
}

template <int dim,int nstate,typename real,typename MeshType>
real ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_boundary_layer_thickness(){
    real boundary_layer_thickness;
    std::array<real,dim+2> ns_soln_of_sampling;
    for(int i=0;i<number_of_sampling;++i){
        if constexpr(nstate==dim+2){
            ns_soln_of_sampling = soln_of_sampling[i];
        } else if constexpr(nstate==dim+3){
            ns_soln_of_sampling = rans_sa_neg_real->extract_rans_conservative_solution(soln_of_sampling[i]);
        } else {
            std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
            std::abort();
        }
        real edge_velocity = navier_stokes_real.compute_velocities(ns_soln_of_sampling).norm();
        if(edge_velocity >= 0.99*U_inf) {
            //boundary_layer_thickness = sqrt((coord_of_sampling[i][0]-start_point[0])*(coord_of_sampling[i][0]-start_point[0])
            //                               +(coord_of_sampling[i][1]-start_point[1])*(coord_of_sampling[i][1]-start_point[1]));
            boundary_layer_thickness = start_point.distance(coord_of_sampling[i]);
            return boundary_layer_thickness;
        }
    }
    std::cout << "ERROR: Fail to capture boundary layer thickness..." << std::endl;
    std::abort();
}

template <int dim,int nstate,typename real,typename MeshType>
void ExtractionFunctional<dim,nstate,real,MeshType>
::evaluate_converged_free_stream_value(){
    const real tolerance = 1e-3;
    std::array<real,dim+2> ns_soln_of_sampling;
    if constexpr(nstate==dim+2){
        ns_soln_of_sampling = soln_of_sampling[0];
    } else if constexpr(nstate==dim+3){
        ns_soln_of_sampling = rans_sa_neg_real->extract_rans_conservative_solution(soln_of_sampling[0]);
    } else {
        std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
        std::abort();
    }
    real U_sampling = navier_stokes_real.compute_velocities(ns_soln_of_sampling).norm();
    real U_sampling_n;
    for(int i=1;i<number_of_sampling;++i){
        if constexpr(nstate==dim+2){
            ns_soln_of_sampling = soln_of_sampling[i];
        } else if constexpr(nstate==dim+3){
            ns_soln_of_sampling = rans_sa_neg_real->extract_rans_conservative_solution(soln_of_sampling[i]);
        } else {
            std::cout << "ERROR: Extraction functional only support nstate == dim+2 or dim+3..." << std::endl;
            std::abort();
        }
        U_sampling_n = navier_stokes_real.compute_velocities(ns_soln_of_sampling).norm();
        if(std::abs(U_sampling-U_sampling_n)<=tolerance){
            U_inf = U_sampling_n;
            density_inf = ns_soln_of_sampling[0];
            std::cout << "Captured converged free stream values..." << std::endl;
            break;
        } else {
            U_sampling = U_sampling_n;
        }
    }
    std::cout << "ERROR: Fail to capture converged free stream values..." << std::endl;
    std::abort();
}

#if PHILIP_DIM!=1
template class ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ExtractionFunctional <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace