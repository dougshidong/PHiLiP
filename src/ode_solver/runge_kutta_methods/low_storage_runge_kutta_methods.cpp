#include "low_storage_runge_kutta_methods.h"

namespace PHiLiP {
namespace ODE {

//##################################################################

// RK4(3)5[3S*]

template <int dim, typename real, typename MeshType>
void RK4_3_5_3SStar<dim,real,MeshType> :: set_gamma()
{
    const double gamma[6][3] = {{0.0, 0.0, 0.0}, 
                                {0.0, 1.0, 0.0},
                                {-0.497531095840104, 1.384996869124138, 0.0}, 
                                {1.010070514199942, 3.878155713328178, 0.0}, 
                                {-3.196559004608766,-2.324512951813145, 1.642598936063715}, 
                                {1.717835630267259, -0.514633322274467, 0.188295940828347}};
    for (int i = 0; i < 6; i++){
        for (int j = 0; j < 3; j++){
            this->butcher_tableau_gamma[i][j] = gamma[i][j];
        }
    }
}

template <int dim, typename real, typename MeshType>
void RK4_3_5_3SStar<dim,real,MeshType> :: set_beta()
{
    const double beta[6] = {0.0, 0.075152045700771, 0.211361016946069, 1.100713347634329, 0.728537814675568, 0.393172889823198};
    this->butcher_tableau_beta.fill(beta);
}

template <int dim, typename real, typename MeshType>
void RK4_3_5_3SStar<dim,real,MeshType> :: set_delta()
{
    const double delta[7] = {1.0, 0.081252332929194, -1.083849060586449, -1.096110881845602, 2.859440022030827, -0.655568367959557, -0.194421504490852};
    this->butcher_tableau_delta.fill(delta);
    
}

template <int dim, typename real, typename MeshType>
void RK4_3_5_3SStar<dim,real,MeshType> :: set_b_hat()
{
    const double b_hat[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    this->butcher_tableau_b_hat.fill(b_hat);
}

//##################################################################

// RK3(2)5F[3S*+]

template <int dim, typename real, typename MeshType>
void RK3_2_5F_3SStarPlus<dim,real,MeshType> :: set_gamma()
{
    const double gamma[6][3] = {{0.0, 0.0, 0.0}, // Ignored
                                {0.0, 1.0, 0.0}, // first loop
                                {0.2587771979725733308135192812685323706, 0.5528354909301389892439698870483746541, 0.0}, 
                                {-0.1324380360140723382965420909764953437, 0.6731871608203061824849561782794643600, 0.0}, 
                                {0.05056033948190826045833606441415585735, 0.2803103963297672407841316576323901761, 0.2752563273304676380891217287572780582}, 
                                {0.5670532000739313812633197158607642990, 0.5521525447020610386070346724931300367, -0.8950526174674033822276061734289327568}};
    for (int i = 0; i < 6; i++){
        for (int j = 0; j < 3; j++){
            this->butcher_tableau_gamma[i][j] = gamma[i][j];
        }
    }
}

template <int dim, typename real, typename MeshType>
void RK3_2_5F_3SStarPlus<dim,real,MeshType> :: set_beta()
{
    const double beta[6] = {0.0, 0.11479359710235412, 0.089334428531133159, 0.43558710250086169, 0.24735761882014512, 0.11292725304550591};
    this->butcher_tableau_beta.fill(beta);
}

template <int dim, typename real, typename MeshType>
void RK3_2_5F_3SStarPlus<dim,real,MeshType> :: set_delta()
{
    const double delta[5] = {1.0, 0.34076558793345252, 0.34143826550033862, 0.72292753667879872, 0.0};
    this->butcher_tableau_delta.fill(delta);
    
}

template <int dim, typename real, typename MeshType>
void RK3_2_5F_3SStarPlus<dim,real,MeshType> :: set_b_hat()
{
    const double b_hat[6] = {0.094841667050357029, 0.17263713394303537, 0.39982431890843712, 0.17180168075801786, 0.058819144221557401, 0.1020760551185952388626787099944507877};
    this->butcher_tableau_b_hat.fill(b_hat);   
}

//##################################################################
template class RK4_3_5_3SStar<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RK4_3_5_3SStar<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RK4_3_5_3SStar<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

template class RK3_2_5F_3SStarPlus<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RK3_2_5F_3SStarPlus<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RK3_2_5F_3SStarPlus<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

/*
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
*/

} // ODESolver namespace
} // PHiLiP namespace
