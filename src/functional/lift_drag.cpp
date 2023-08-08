#include "lift_drag.hpp"

namespace PHiLiP {

template <int dim,int nstate,typename real,typename MeshType>
LiftDragFunctional<dim,nstate,real,MeshType>
::LiftDragFunctional(
    std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
    const Functional_types functional_type)
    : Functional<dim,nstate,real,MeshType>(dg_input)
    , functional_type(functional_type)
    , angle_of_attack(dg_input->all_parameters->euler_param.angle_of_attack)
    , rotation_matrix(initialize_rotation_matrix(this->angle_of_attack))
    , lift_vector(initialize_lift_vector(this->rotation_matrix))
    , drag_vector(initialize_drag_vector(this->rotation_matrix))
    , force_dimensionalization_factor(this->initialize_force_dimensionalization_factor())
{   
    switch(functional_type) {
        case Functional_types::lift : this->force_vector = lift_vector; break;
        case Functional_types::drag : this->force_vector = drag_vector; break;
        default: break;
    }
}

template <int dim,int nstate,typename real,typename MeshType>
double LiftDragFunctional<dim,nstate,real,MeshType>
::initialize_force_dimensionalization_factor()
{
    const double ref_length = this->euler_fad_fad->ref_length;
    const double dynamic_pressure_inf = this->euler_fad_fad->dynamic_pressure_inf;

    return 1.0 / (ref_length * dynamic_pressure_inf);
}

template <int dim,int nstate,typename real,typename MeshType>
dealii::Tensor<2,dim,double> LiftDragFunctional<dim,nstate,real,MeshType>
::initialize_rotation_matrix(const double angle_of_attack)
{
    dealii::Tensor<2,dim,double> rotation_matrix;
    if constexpr (dim == 1) {
        assert(false);
    }

    rotation_matrix[0][0] = cos(angle_of_attack);
    rotation_matrix[0][1] = -sin(angle_of_attack);
    rotation_matrix[1][0] = sin(angle_of_attack);
    rotation_matrix[1][1] = cos(angle_of_attack);

    if constexpr (dim == 3) {
        rotation_matrix[0][2] = 0.0;
        rotation_matrix[1][2] = 0.0;

        rotation_matrix[2][0] = 0.0;
        rotation_matrix[2][1] = 0.0;
        rotation_matrix[2][2] = 1.0;
    }

    return rotation_matrix;
}

template <int dim,int nstate,typename real,typename MeshType>
dealii::Tensor<1,dim,double> LiftDragFunctional<dim,nstate,real,MeshType>
::initialize_lift_vector (const dealii::Tensor<2,dim,double> &rotation_matrix)
{
    dealii::Tensor<1,dim,double> lift_direction;
    lift_direction[0] = 0.0;
    lift_direction[1] = 1.0;

    if constexpr (dim == 1) {
        assert(false);
    }
    if constexpr (dim == 3) {
        lift_direction[2] = 0.0;
    }

    dealii::Tensor<1,dim,double> vec;
    vec = rotation_matrix * lift_direction;

    return vec;
}

template <int dim,int nstate,typename real,typename MeshType>
dealii::Tensor<1,dim,double> LiftDragFunctional<dim,nstate,real,MeshType>
::initialize_drag_vector (const dealii::Tensor<2,dim,double> &rotation_matrix)
{
    dealii::Tensor<1,dim,double> drag_direction;
    drag_direction[0] = 1.0;
    drag_direction[1] = 0.0;

    if constexpr (dim == 1) {
        assert(false);
    }
    if constexpr (dim == 3) {
        drag_direction[2] = 0.0;
    }

    dealii::Tensor<1,dim,double> vec;
    vec = rotation_matrix * drag_direction;

    return vec;
}

template <int dim,int nstate,typename real,typename MeshType>
real LiftDragFunctional<dim,nstate,real,MeshType>
::evaluate_functional(const bool compute_dIdW, 
                      const bool compute_dIdX, 
                      const bool compute_d2I)
{
    double value = Functional<dim,nstate,real,MeshType>::evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);

    if (functional_type == Functional_types::lift) {
        this->pcout << "Lift value: " << value << "\n";
        //std::cout << "Lift value: " << value << std::cout;
        //std::cout << "Lift value: " << value << std::cout;
    }
    if (functional_type == Functional_types::drag) {
        this->pcout << "Drag value: " << value << "\n";
    }

    return value;
}

#if PHILIP_DIM!=1
template class LiftDragFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace

