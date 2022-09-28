#include "runge_kutta_methods.h"

namespace PHiLiP {
namespace ODE {

//##################################################################
template <int dim, typename real, typename MeshType>
void SSPRK3Explicit<dim,real,MeshType> :: set_a()
{
    const double butcher_tableau_a_values[9] = {0,0,0,1.0,0,0,0.25,0.25,0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void SSPRK3Explicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void SSPRK3Explicit<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[3] = {0,1.0,0.5};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}

//##################################################################
template <int dim, typename real, typename MeshType>
void RK4Explicit<dim,real,MeshType> :: set_a()
{
    const double butcher_tableau_a_values[16] = {0,0,0,0,0.5,0,0,0,0,0.5,0,0,0,0,1.0,0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void RK4Explicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[4] = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void RK4Explicit<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[4] = {0,0.5,0.5,1.0};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}

//##################################################################
template <int dim, typename real, typename MeshType>
void EulerExplicit<dim,real,MeshType> :: set_a()
{
    const double butcher_tableau_a_values[1] = {0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void EulerExplicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[1] = {1.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void EulerExplicit<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[1] = {0};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}

//##################################################################
template <int dim, typename real, typename MeshType>
void EulerImplicit<dim,real,MeshType> :: set_a()
{
    const double butcher_tableau_a_values[1] = {1.0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void EulerImplicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[1] = {1.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void EulerImplicit<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[1] = {1.0};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}

//##################################################################
template <int dim, typename real, typename MeshType>
void DIRK2Implicit<dim,real,MeshType> :: set_a()
{
    // Pareschi & Russo DIRK, x = 1 - sqrt(2)/2
    // see: wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Diagonally_Implicit_Runge%E2%80%93Kutta_methods
    const double x = 1.0 - sqrt(2.0)/2.0;
    const double butcher_tableau_a_values[4] = {x,0,(1-2*x),x};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void DIRK2Implicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[2] = {0.5, 0.5};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void DIRK2Implicit<dim,real,MeshType> :: set_c()
{
    const double x = 1.0 - sqrt(2.0)/2.0;
    const double butcher_tableau_c_values[2] = {x, 1.0-x};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}

//##################################################################
template class SSPRK3Explicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class SSPRK3Explicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class SSPRK3Explicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

template class RK4Explicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RK4Explicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RK4Explicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

template class EulerExplicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class EulerExplicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class EulerExplicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

template class EulerImplicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class EulerImplicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    // Commenting higher dimensions as they have not been tested yet
    //template class EulerImplicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

template class DIRK2Implicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class DIRK2Implicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    // Commenting higher dimensions as they have not been tested yet
    //template class DIRK2Implicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
