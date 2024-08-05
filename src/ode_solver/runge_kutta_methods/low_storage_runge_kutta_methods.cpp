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
    const double beta[6] = {0.0, 0.2300298624518076223899418286314123354, 0.3021434166948288809034402119555380003, 
                            0.8025606185416310937583009085873554681, 0.4362158943603440930655148245148766471, 0.11292725304550591};
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
    const double b_hat[6] = {0.094841667050357029, 0.17263713394303537, 0.39982431890843712, 0.17180168075801786, 0.058819144221557401, 
                             0.1020760551185952388626787099944507877};
    this->butcher_tableau_b_hat.fill(b_hat);   
}

//##################################################################

// RK4(3)9F[3S*+]

template <int dim, typename real, typename MeshType>
void RK4_3_9F_3SStarPlus<dim,real,MeshType> :: set_gamma()
{
    const double gamma[10][3] = {{0.0, 0.0, 0.0}, // Ignored
                                {0.0, 1.0, 0.0}, // first loop
                                {-4.655641447335068552684422206224169103, 2.499262792574495009336242992898153462, 0.0}, 
                                {-0.7720265099645871829248487209517314217, 0.5866820377718875577451517985847920081, 0.0}, 
                                {-4.024436690519806086742256154738379161, 1.205146086523094569925592464380295241, 0.7621006678721315291614677352949377871}, 
                                {-0.02129676284018530966221583708648634733, 0.3474793722186732780030762737753849272, -0.1981182504339400567765766904309673119},
                                {-2.435022509790109546199372365866450709, 1.321346060965113109321230804210670518, -0.6228959218699007450469629366684127462},
                                {0.01985627297131987000579523283542615256, 0.3119636464694193615946633676950358444, -0.3752248380775956442989480369774937099},
                                {-0.2810791146791038566946663374735713961, 0.4351419539684379261368971206040518552, -0.3355438309135169811915662336248989661},
                                {0.1689434168754859644351230590422137972, 0.2359698130028753572503744518147537768, -0.04560955005031121479972862973705108039}};
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 3; j++){
            this->butcher_tableau_gamma[i][j] = gamma[i][j];
        }
    }
}

template <int dim, typename real, typename MeshType>
void RK4_3_9F_3SStarPlus<dim,real,MeshType> :: set_beta()
{
    const double beta[10] = {0.0, 0.2836343005184365275160654678626695428, 0.9736500104654741223716056170419660217, 0.3382359225242515288768487569778320563, 
                            -0.3584943611106183357043212309791897386, -0.004113944068471528211627210454497620358, 1.427968894048586363415504654313371031,
                            0.01808470948394314017665968411915568633, 0.1605770645946802213926893453819236685, 0.2952227015964591648775833803635147962};
    this->butcher_tableau_beta.fill(beta);
}

template <int dim, typename real, typename MeshType>
void RK4_3_9F_3SStarPlus<dim,real,MeshType> :: set_delta()
{
    const double delta[9] = {1.0, 1.262923876648114432874834923838556100, 0.7574967189685911558308119415539596711, 0.5163589453140728104667573195005629833, 
    -0.02746327421802609557034437892013640319, -0.4382673178127944142238606608356542890,  1.273587294602656522645691372699677063, 
    -0.6294740283927400326554066998751383342, 0.0};
    this->butcher_tableau_delta.fill(delta);
    
}

template <int dim, typename real, typename MeshType>
void RK4_3_9F_3SStarPlus<dim,real,MeshType> :: set_b_hat()
{
    const double b_hat[10] = {0.02483675912451591196775756814283216443, 0.1866327774562103796990092260942180726, 0.05671080795936984495604436622517631183, 
                             -0.003447695439149287702616943808570747099, 0.003602245056516636472203469198006404016, 0.4545570622145088936800484247980581766,
                             -0.0002434665289427612407531544765622888855, 0.06642755361103549971517945063138312147, 0.1613697079523505006226025497715177578,
                              0.04955424859358438183052504342394102722};
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

template class RK4_3_9F_3SStarPlus<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RK4_3_9F_3SStarPlus<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RK4_3_9F_3SStarPlus<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
