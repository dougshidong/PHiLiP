#ifndef __LOW_STORAGE_RUNGE_KUTTA_METHODS__
#define __LOW_STORAGE_RUNGE_KUTTA_METHODS__

#include "low_storage_rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {


/// Three-register method that include an error estimate
/** see 
 *  David Ketcheson. "Runge-Kutta Methods with Minimum Storage Implementations" Journal of computational physics 229.5 (2010): 1763-1773. 
 * 
 *  Naming convention: RK4 with an embedded RK3 method, 5 stages, 3S* method*/

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


/// Three-register method that require a fourth register for the error estimate
/** see 
 *  Hedrik Ranocha, Lisandro Dalcin, Matteo Parsani, David Ketcheson. 
 *  "Optimized Runge-Kutta Methods with Automatic Step Size Control for Compressible Computational Fluid Dynamics" Communications on Applied Mathematics and Computation Volume 4 (2022): 1191-1228. 
 *  https://github.com/ranocha/Optimized-RK-CFD 
 * 
 * Naming convention: RK3 with an embedded RK2 method, 6 stages FSAL, 3S*+ method*/
 
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

/// Three-register method that require a fourth register for the error estimate
/** see 
 *  Hedrik Ranocha, Lisandro Dalcin, Matteo Parsani, David Ketcheson. 
 *  "Optimized Runge-Kutta Methods with Automatic Step Size Control for Compressible Computational Fluid Dynamics" Communications on Applied Mathematics and Computation Volume 4 (2022): 1191-1228. 
 *  https://github.com/ranocha/Optimized-RK-CFD 
 * 
 * 
 * Naming convention: RK4 with an embedded RK3 method, 10 stages FSAL, 3S*+ method*/
 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RK4_3_9F_3SStarPlus: public LowStorageRKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    RK4_3_9F_3SStarPlus(const int n_rk_stages, const int num_delta, const std::string rk_method_string_input) 
        : LowStorageRKTableauBase<dim,real,MeshType>(n_rk_stages, num_delta, rk_method_string_input) { }

protected:
    /// Setters
    void set_gamma() override;
    void set_beta() override;
    void set_delta() override;
    void set_b_hat() override;
};

/// Three-register method that require a fourth register for the error estimate
/** see 
 *  Hedrik Ranocha, Lisandro Dalcin, Matteo Parsani, David Ketcheson. 
 *  "Optimized Runge-Kutta Methods with Automatic Step Size Control for Compressible Computational Fluid Dynamics" Communications on Applied Mathematics and Computation Volume 4 (2022): 1191-1228. 
 *  https://github.com/ranocha/Optimized-RK-CFD 
 * 
 *  
 *  Naming convention: RK5 with an embedded RK4 method, 11 stages FSAL, 3S*+ method*/
 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RK5_4_10F_3SStarPlus: public LowStorageRKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    RK5_4_10F_3SStarPlus(const int n_rk_stages, const int num_delta, const std::string rk_method_string_input) 
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
