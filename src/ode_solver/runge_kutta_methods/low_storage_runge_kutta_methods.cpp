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

// RK5(4)10F[3S*+]

template <int dim, typename real, typename MeshType>
void RK5_4_10F_3SStarPlus<dim,real,MeshType> :: set_gamma()
{
    const double gamma[11][3] = {{0.0, 0.0, 0.0}, // Ignored
                                {0.0, 1.0, 0.0}, // first loop
                                {0.4043660121685749695640462197806189975, 0.6871467028161416909922221357014564412, 0.0}, 
                                {-0.8503427289575839690883191973980814832, 1.093024748914750833700799552463885117, 0.0}, 
                                {-6.950894175262117526410215315179482885, 3.225975379607193001678365742708874597, -2.393405133244194727221124311276648940}, 
                                {0.9238765192731084931855438934978371889, 1.041153702510101386914019859778740444, -1.902854422421760920850597670305403139},
                                {-2.563178056509891340215942413817786020, 1.292821487912164945157744726076279306, -2.820042207399977261483046412236557428},
                                {0.2545744879365226143946122067064118430, 0.7391462755788122847651304143259254381, -1.832698465277380999601896111079977378},
                                {0.3125831707411998258746812355492206137, 0.1239129251371800313941948224441873274, -0.2199094483084671192328083958346519535},
                                {-0.7007114414440507927791249989236719346, 0.1842753472370123193132193302369345580, -0.4082430635847870963724591602173546218},
                                {0.4839621016023833375810172323297465039, 0.05712788998796583446479387686662738843, -0.1377669797880289713535665985132703979}};
    for (int i = 0; i < 11; i++){
        for (int j = 0; j < 3; j++){
            this->butcher_tableau_gamma[i][j] = gamma[i][j];
        }
    }
}

template <int dim, typename real, typename MeshType>
void RK5_4_10F_3SStarPlus<dim,real,MeshType> :: set_beta()
{
    const double beta[11] = {0.0, 0.2597883554788674084039539165398464630, 0.01777008889438867858759149597539211023, 0.2481636629715501931294746189266601496,
                            0.7941736871152005775821844297293296135, 0.3885391285642019129575902994397298066, 0.1455051657916305055730603387469193768, 
                            0.1587517385964749337690916959584348979, 0.1650605617880053419242434594242509601, 0.2118093284937153836908655490906875007,
                            0.1559392342362059886106995325687547506};
    this->butcher_tableau_beta.fill(beta);
}

template <int dim, typename real, typename MeshType>
void RK5_4_10F_3SStarPlus<dim,real,MeshType> :: set_delta()
{
    const double delta[10] = {1.0, -0.1331778419508803397033287009506932673, 0.8260422814750207498262063505871077303,
                            1.513700425755728332485300719652378197, -1.305810059935023735972298885749903694, 3.036678802924163246003321318996156380,
                            -1.449458274398895177922690618003584514, 3.834313899176362315089976408899373409, 4.122293760012985409330881631526514714,
                            0.0};
    this->butcher_tableau_delta.fill(delta);
    
}

template <int dim, typename real, typename MeshType>
void RK5_4_10F_3SStarPlus<dim,real,MeshType> :: set_b_hat()
{
    const double b_hat[11] = {-0.02019255440012066080909442770590267512, 0.02737903480959184339932730854141598275, 0.3028818636145965534365173822296811090, 
                             -0.03656843880622222190071445247906780540, 0.3982664774676767729863101188528827405, -0.05715959421140685436681459970502471634,
                             0.09849855103848558320961101178888983150, 0.06654601552456084978615342374581437947, 0.09073479542748112726465375642050504556,
                             0.08432289325330803924891866923939606351, 0.04529095628204896774513180907141004447};
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

template class RK5_4_10F_3SStarPlus<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RK5_4_10F_3SStarPlus<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RK5_4_10F_3SStarPlus<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
