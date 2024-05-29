#ifndef __RUNGE_KUTTA_METHODS__
#define __RUNGE_KUTTA_METHODS__

#include "rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

/// Third-order strong stability preserving explicit RK
/** see 
 *  Shu, Chi-Wang, and Stanley Osher. "Efficient implementation of essentially non-oscillatory shock-capturing schemes." Journal of computational physics 77.2 (1988): 439-471. */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class SSPRK3Explicit: public RKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    SSPRK3Explicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

/// Classical fourth-order explicit RK
/** See
 * https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods
 * Section titled "Classic fourth-order method" */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RK4Explicit: public RKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    RK4Explicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

/// Forward Euler (explicit) 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EulerExplicit: public RKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    EulerExplicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};


///  Heun's method (explicit trapezoid rule; SSP 2,2)
/** See
 * https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods
 * Section titled "Heun's method" */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class HeunExplicit: public RKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    HeunExplicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};


/// Implicit Euler 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EulerImplicit: public RKTableauBase <dim, real, MeshType>
{
public:
    ///Constructor
    EulerImplicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

/// Second-order diagonally-implicit RK
/// two-stage, stiffly-accurate, L-stable SDIRK, gamma = (2 - sqrt(2))/2
/// see "Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review"
/// Kennedy & Carpenter, 2016
/// Sec. 4.1.2
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DIRK2Implicit: public RKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    DIRK2Implicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

/// Third-order diagonally-implicit RK
/// three-stage, stiffly-accurate SDIRK, gamma = 0.43586652150845899941601945
/// see "Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review"
/// Kennedy & Carpenter, 2016
/// Sec. 5.1.3
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DIRK3Implicit: public RKTableauBase <dim, real, MeshType>
{
public:
    /// Constructor
    DIRK3Implicit(const int n_rk_stages, const std::string rk_method_string_input) 
        : RKTableauBase<dim,real,MeshType>(n_rk_stages, rk_method_string_input) { }

protected:

    const double gam = 0.435866521508458999416019; ///< Constant "gamma" given by Kennedy & Carpenter
    const double alpha = 1 - 4*gam + 2*gam*gam; ///< Constant "alpha" given by Kennedy & Carpenter
    const double beta = -1 + 6*gam - 9*gam*gam + 3*gam*gam*gam; ///< Constant "beta" given by Kennedy & Carpenter
    const double b2 = -3*alpha*alpha / 4.0 / beta; ///< Constant "b2" given by Kennedy & Carpenter
    const double c2 = (2 - 9*gam + 6*gam*gam) / 3.0 / alpha; ///< Constant "c2" given by Kennedy & Carpenter

    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

} // ODE namespace
} // PHiLiP namespace

#endif
