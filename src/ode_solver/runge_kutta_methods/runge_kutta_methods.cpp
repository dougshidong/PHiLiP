#include "runge_kutta_methods.h"

namespace PHiLiP {
namespace ODE {

//##################################################################
template <int dim, typename real, typename MeshType>
SSPRK3Explicit<dim,real,MeshType> :: SSPRK3Explicit(int n_rk_stages)
        : RKTableauBase<dim,real,MeshType> ()
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    // RKSSP3 (RK-3 Strong-Stability-Preserving)
    const double butcher_tableau_a_values[9] = {0,0,0,1.0,0,0,0.25,0.25,0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
    const double butcher_tableau_b_values[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
    this->pcout << "Assigned RK method: 3rd order SSP (explicit)" << std::endl;
}

//##################################################################
template <int dim, typename real, typename MeshType>
RK4Explicit<dim,real,MeshType> :: RK4Explicit(int n_rk_stages)
        : RKTableauBase<dim,real,MeshType> ()
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    const double butcher_tableau_a_values[16] = {0,0,0,0,0.5,0,0,0,0,0.5,0,0,0,0,1.0,0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
    const double butcher_tableau_b_values[4] = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
    this->pcout << "Assigned RK method: 4th order classical RK (explicit)" << std::endl;
}

//##################################################################
template <int dim, typename real, typename MeshType>
EulerExplicit<dim,real,MeshType> :: EulerExplicit(int n_rk_stages)
        : RKTableauBase<dim,real,MeshType> ()
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    const double butcher_tableau_a_values[1] = {0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
    const double butcher_tableau_b_values[1] = {1.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
    this->pcout << "Assigned RK method: Forward Euler (explicit)" << std::endl;
}

//##################################################################
template <int dim, typename real, typename MeshType>
EulerImplicit<dim,real,MeshType> :: EulerImplicit(int n_rk_stages)
        : RKTableauBase<dim,real,MeshType> ()
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    const double butcher_tableau_a_values[1] = {1.0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
    const double butcher_tableau_b_values[1] = {1.0};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
    this->pcout << "Assigned RK method: Implicit Euler (implicit)" << std::endl;
}

//##################################################################
template <int dim, typename real, typename MeshType>
DIRK2Implicit<dim,real,MeshType> :: DIRK2Implicit(int n_rk_stages)
        : RKTableauBase<dim,real,MeshType> ()
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    // Pareschi & Russo DIRK, x = 1 - sqrt(2)/2
    // see: wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Diagonally_Implicit_Runge%E2%80%93Kutta_methods
    const double x = 0.2928932188134525; //=1-sqrt(2)/2
    const double butcher_tableau_a_values[4] = {x,0,(1-2*x),x};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
    const double butcher_tableau_b_values[2] = {0.5, 0.5};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
    this->pcout << "Assigned RK method: 2nd-order DIRK (implicit)" << std::endl;
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
    template class EulerImplicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

template class DIRK2Implicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class DIRK2Implicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class DIRK2Implicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
