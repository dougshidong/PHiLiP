#include "PERK_methods.h"

namespace PHiLiP {
namespace ODE {

//##################################################################
template <int dim, typename real, typename MeshType>
void PERK_10_2<dim,real,MeshType> :: set_a1()
{
    const double butcher_tableau_a1_values[100] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.05555555555555555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.1666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.2777777777777778, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.2753855669141815, 0.0, 0.0, 0.0, 0.0, 0.05794776641915183, 0.0, 0.0, 0.0, 0.0,
                                                  0.2624172478767012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1264716410121877, 0.0, 0.0, 0.0,
                                                  0.2288001625437463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2156442819006982, 0.0, 0.0,
                                                  0.1533674922301650, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3466325077698350, 0.0};


    this->butcher_tableau_a1.fill(butcher_tableau_a1_values);
}

template <int dim, typename real, typename MeshType>
void PERK_10_2<dim,real,MeshType> :: set_a2()
{
    const double butcher_tableau_a2_values[100] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.05555555555555555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.1666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.2340959502023788, 0.0, 0.0, 0.0, 0.04368182757539894, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.2406628557314545, 0.0, 0.0, 0.0, 0.0, 0.09267047760187887, 0.0, 0.0, 0.0, 0.0,
                                                  0.2361284000098332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1527604888790557, 0.0, 0.0, 0.0,
                                                  0.2114166292436466, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2330278152007978, 0.0, 0.0,
                                                  0.1467310489115189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3532689510884811, 0.0};
    this->butcher_tableau_a2.fill(butcher_tableau_a2_values);
}

template <int dim, typename real, typename MeshType>
void PERK_10_2<dim,real,MeshType> :: set_b()
{
    const double butcher_tableau_b_values[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    this->butcher_tableau_b.fill(butcher_tableau_b_values);
}

template <int dim, typename real, typename MeshType>
void PERK_10_2<dim,real,MeshType> :: set_c()
{
    const double butcher_tableau_c_values[10] = {0, 1/18, 1/9, 1/6, 2/9, 5/18, 1/3, 7/18, 4/9, 1/2};
    this->butcher_tableau_c.fill(butcher_tableau_c_values);
}


//##################################################################
template class PERK_10_2<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class PERK_10_2<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class PERK_10_2<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace