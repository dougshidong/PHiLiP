#ifndef __RUNGE_KUTTA_METHODS__
#define __RUNGE_KUTTA_METHODS__

#include "rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

/// Third-order strong stability preserving explicit RK
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

    /// Destructor
    ~SSPRK3Explicit() {};

protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

/// Classical fourth-order explicit RK
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

    /// Destructor
    ~RK4Explicit() {};

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

    /// Destructor
    ~EulerExplicit() {};
    
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

    /// Destructor
    ~EulerImplicit() {};
    
protected:
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};


/// Second-order diagonally-implicit RK
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

    /// Destructor
    ~DIRK2Implicit() {};
    
protected:
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
