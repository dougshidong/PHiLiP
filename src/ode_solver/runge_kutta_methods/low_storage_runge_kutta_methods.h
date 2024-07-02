#ifndef __LOW_STORAGE_RUNGE_KUTTA_METHODS__
#define __LOW_STORAGE_RUNGE_KUTTA_METHODS__

#include "low_storage_rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RK4_3_5_3SStar: public LowStorageRKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    RK4_3_5_3SStar(const int n_rk_stages, const int num_delta, const std::string rk_method_string_input) 
        : LowStorageRKTableauBase<dim,real,MeshType>(n_rk_stages, num_delta, rk_method_string_input) { }

protected:
    /// Setters
    void set_gamma() override;
    void set_beta() override;
    void set_delta() override;
    void set_b_hat() override;
};


#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RK3_2_5F_3SStarPlus: public LowStorageRKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    RK3_2_5F_3SStarPlus(const int n_rk_stages, const int num_delta, const std::string rk_method_string_input) 
        : LowStorageRKTableauBase<dim,real,MeshType>(n_rk_stages, num_delta, rk_method_string_input) { }

protected:
    /// Setters
    void set_gamma() override;
    void set_beta() override;
    void set_delta() override;
    void set_b_hat() override;
};


} // ODE namespace
} // PHiLiP namespace

#endif
