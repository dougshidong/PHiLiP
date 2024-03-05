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
void HeunExplicit<dim,real,MeshType> :: set_a()
{
    const double butcher_tableau_a_values[4] = {0,0,1.0,0};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void HeunExplicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[2] = {0.5, 0.5};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void HeunExplicit<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[2] = {0,1.0};
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
    // two-stage, stiffly-accurate, L-stable SDIRK, gamma = (2 - sqrt(2))/2
    // see "Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review"
    // Kennedy & Carpenter, 2016
    // Sec. 4.1.2
    const double gam = 1.0 - sqrt(2.0)/2.0;
    const double butcher_tableau_a_values[4] = {gam,0,(1-gam),gam};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void DIRK2Implicit<dim,real,MeshType> :: set_b()
{
    const double gam = 1.0 - sqrt(2.0)/2.0;
    const double butcher_tableau_b_values[2] = {(1-gam), gam};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void DIRK2Implicit<dim,real,MeshType> :: set_c()
{
    const double x = 1.0 - sqrt(2.0)/2.0;
    const double butcher_tableau_c_values[2] = {x, 1.0};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}

//##################################################################
template <int dim, typename real, typename MeshType>
void DIRK3Implicit<dim,real,MeshType> :: set_a()
{
    // three-stage, stiffly-accurate SDIRK, gamma = 0.43586652150845899941601945
    // see "Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review"
    // Kennedy & Carpenter, 2016
    // Sec. 5.1.3
    const double butcher_tableau_a_values[9] = {gam,       0,   0,
                                                (c2-gam),  gam, 0,
                                                (1-b2-gam), b2, gam};
    this->butcher_tableau_a.fill(butcher_tableau_a_values);
}

template <int dim, typename real, typename MeshType>
void DIRK3Implicit<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[3] = {(1-b2-gam), b2, gam};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void DIRK3Implicit<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[3] = {gam, c2, 1};
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

template class HeunExplicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class HeunExplicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class HeunExplicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
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

template class DIRK3Implicit<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class DIRK3Implicit<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class DIRK3Implicit<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
