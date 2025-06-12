#ifndef __PERK_METHODS__
#define __PERK_METHODS__

#include "PERK_tableau_base.h"

namespace PHiLiP {
namespace ODE {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PERK_10_2: public PERKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    PERK_10_2(const int n_rk_stages, const std::string rk_method_string_input) 
        : PERKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a1
    void set_a1() override;

    /// Setter for butcher_tableau_a2
    void set_a2() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};
} // ODE namespace
} // PHiLiP namespace

#endif