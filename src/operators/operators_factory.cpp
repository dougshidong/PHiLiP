#include <deal.II/base/geometry_info.h>
#include "operators_factory.hpp"
#include "operators.h"

namespace PHiLiP {
namespace OPERATOR {

template <int dim, typename real>
std::shared_ptr< OperatorsBase<dim,real,2*dim> >//returns type OperatorsBase
OperatorsFactory<dim,real>
::create_operators(
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input,//number of states input
        const unsigned int /*degree*/,//degree not really needed at the moment
        const unsigned int max_degree_input,//max poly degree for operators
        const unsigned int grid_degree_input)//max grid degree for operators
{

//    const unsigned int n_dofs_max = pow(max_degree_input + 1, dim);
//    const unsigned int overintegration = parameters_input->overintegration;
//    const unsigned int n_quad_pts_vol = pow(max_degree_input + 1 + overintegration, dim);
//    const unsigned int n_quad_pts_face = pow(max_degree_input + 1 + overintegration, dim-1);
//    const unsigned int n_dofs_max_metric = pow(grid_degree_input + 1, dim);
    const unsigned int n_faces = dealii::GeometryInfo<dim>::faces_per_cell;
    if(n_faces == 2*dim){
        if(nstate_input == 1){
            return std::make_shared< OperatorsBaseState<dim,real,1,2*dim> >(parameters_input, max_degree_input, grid_degree_input);//, degree, max_degree_input, grid_degree_input);
        }
        else if(nstate_input == 2){
            return std::make_shared< OperatorsBaseState<dim,real,2,2*dim> >(parameters_input, max_degree_input, grid_degree_input);//, degree, max_degree_input, grid_degree_input);
        }
        else if(nstate_input == 3){
            return std::make_shared< OperatorsBaseState<dim,real,3,2*dim> >(parameters_input, max_degree_input, grid_degree_input);//, degree, max_degree_input, grid_degree_input);
        }
        else if(nstate_input == 4){
            return std::make_shared< OperatorsBaseState<dim,real,4,2*dim> >(parameters_input, max_degree_input, grid_degree_input);//, degree, max_degree_input, grid_degree_input);
        }
        else if(nstate_input == 5){
            return std::make_shared< OperatorsBaseState<dim,real,5,2*dim> >(parameters_input, max_degree_input, grid_degree_input);//, degree, max_degree_input, grid_degree_input);
        }
        else{
            std::cout<<"Number of states "<<nstate_input<<"not supported."<<std::endl;
            return nullptr;
        }
    }
    else{
        std::cout<<"Only Hex. elements are supports."<<std::endl;
        return nullptr;
    }
   //try without make shared for now
   // return OperatorsBaseState<dim,real,nstate_input,max_degree_input,grid_degree_input,ndofs_max,n_quad_pts_vol,n_quad_pts_face,n_dofs_max_metric> (parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
//    return std::make_shared< OperatorsVolume<dim,real,nstate_input,degree,ndofs_max,n_quad_pts_vol> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
//    return std::make_shared< OperatorsSurface<dim,real,nstate_input,degree,ndofs_max,n_quad_pts_face> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
//    return std::make_shared< OperatorsMetric<dim,real,nstate_input,degree,ndofs_max_metric,n_quad_pts_vol,n_quad_pts_face> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
}

template class OperatorsFactory <PHILIP_DIM, double>;

} // OPERATOR namespace
} // PHiLiP namespace
